//! `run_cnv` on one GSE148673 sample (Gao et al., CopyKat, 2021).
//!
//! ```bash
//! cargo run --release --example gse148673 -- <sample.txt> <genes.tsv> <hg19|hg38>
//! ```
//!
//! `sample.txt` is a decompressed `*_combined_UMIcount_CellTypes_*.txt`:
//! a barcode header, rows `copykat.pred` and `cluster.pred` (T/N), then one
//! row of UMI counts per gene symbol. `genes.tsv` is
//! `symbol chrom start end biotype` without header: GENCODE v19 for the
//! hg19 samples (DCIS1, TNBC1-3), Ensembl 93 for the rest. Only symbols on
//! chromosomes 1 to 22 are kept; a duplicated symbol keeps its first locus.
//! The build picks the exclusion intervals (`EXCLUDE_HG19` or `EXCLUDE_HG38`).
//!
//! The copykat `N` cells are the reference. With `CNV_POST=<file>` the
//! posteriors are written as little-endian `f32`: `mu`, `sigma`, then
//! `log_odds`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use cnv_sc_rs::cnv::{CnvParams, run_cnv};
use cnv_sc_rs::genome::{CsrCounts, EXCLUDE_HG19, EXCLUDE_HG38};
use cnv_sc_rs::prelude::Verbosity;
use cnv_sc_rs::reference::Reference;

/// Autosomal loci by symbol, first occurrence wins.
///
/// ### Params
///
/// * `path` - Annotation TSV
///
/// ### Returns
///
/// `symbol -> (chrom, start, end)`.
fn read_genes(path: &str) -> HashMap<String, (u32, u64, u64)> {
    let mut out = HashMap::new();
    let f = BufReader::new(std::fs::File::open(path).expect("open genes"));
    for line in f.lines() {
        let line = line.expect("read genes");
        let mut it = line.split('\t');
        let (sym, chr, s, e) = (
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
            it.next().unwrap(),
        );
        let Ok(chrom) = chr.trim_start_matches("chr").parse::<u32>() else {
            continue;
        };
        if (1..=22).contains(&chrom) {
            out.entry(sym.to_string())
                .or_insert((chrom, s.parse().unwrap(), e.parse().unwrap()));
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (sample, genes_path) = (&args[1], &args[2]);
    let exclude = match args[3].as_str() {
        "hg19" => EXCLUDE_HG19.to_vec(),
        "hg38" => EXCLUDE_HG38.to_vec(),
        other => panic!("build must be hg19 or hg38, got {other}"),
    };
    let t = Instant::now();
    let loci = read_genes(genes_path);

    let mut lines = BufReader::new(std::fs::File::open(sample).expect("open sample")).lines();
    let n_cells = lines.next().unwrap().unwrap().split('\t').count();
    let pred: Vec<bool> = lines
        .next()
        .unwrap()
        .unwrap()
        .split('\t')
        .skip(1)
        .map(|s| s == "T")
        .collect();
    assert_eq!(pred.len(), n_cells, "copykat.pred length");
    lines.next();

    // gene-major triplets, transposed to cell-major CSR below
    let (mut chrom, mut start, mut end) = (Vec::new(), Vec::new(), Vec::new());
    let mut per_cell: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_cells];
    let mut n_rows = 0;
    for line in lines {
        let line = line.unwrap();
        n_rows += 1;
        let mut it = line.split('\t');
        let Some(&(c, s, e)) = loci.get(it.next().unwrap()) else {
            continue;
        };
        let g = chrom.len() as u32;
        let mut any = false;
        for (cell, v) in it.enumerate() {
            if v != "0" {
                per_cell[cell].push((g, v.parse().unwrap()));
                any = true;
            }
        }
        if any {
            chrom.push(c);
            start.push(s);
            end.push(e);
        }
    }
    let n_genes = chrom.len();
    let (mut indptr, mut indices, mut data) = (vec![0u32], Vec::new(), Vec::new());
    for r in &per_cell {
        indices.extend(r.iter().map(|x| x.0));
        data.extend(r.iter().map(|x| x.1));
        indptr.push(indices.len() as u32);
    }
    drop(per_cell);
    let normal: Vec<u32> = (0..n_cells as u32).filter(|&i| !pred[i as usize]).collect();
    println!(
        "{n_cells} cells ({} N), {n_rows} rows, {n_genes} autosomal expressed genes, nnz {} ({:.1}%), loaded in {:.1}s",
        normal.len(),
        data.len(),
        100.0 * data.len() as f64 / (n_cells * n_genes) as f64,
        t.elapsed().as_secs_f64()
    );

    let counts = CsrCounts::new(&indptr, &indices, &data, n_genes).unwrap();
    let reference = Reference::<f32>::Cells(&normal);
    let t = Instant::now();
    let res = run_cnv::<f32>(
        &counts,
        &chrom,
        &start,
        &end,
        &reference,
        Some(CnvParams {
            exclude,
            ..CnvParams::default()
        }),
        Verbosity::Normal,
    )
    .unwrap();
    println!(
        "total {:.2}s on {} threads",
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );

    match (res.clones.as_ref(), res.posteriors.as_ref()) {
        (Some(cl), Some(post)) => {
            let ok = res
                .cells
                .iter()
                .enumerate()
                .filter(|&(row, &cell)| (cl.p_tumour[row] > 0.5) == pred[cell as usize])
                .count();
            println!(
                "segments {} ({} non-neutral), agreement with copykat.pred {:.3}",
                res.consensus.len(),
                post.segs.len(),
                ok as f64 / res.cells.len() as f64
            );
            if let Ok(path) = std::env::var("CNV_POST") {
                let mut f = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
                for v in post.mu.iter().chain(&post.sigma).chain(&post.log_odds) {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
                println!("posteriors written to {path}");
            }
        }
        _ => println!("no CNV called"),
    }
}
