//! End-to-end `run_cnv` on simulated counts: stage split, wall-clock and peak
//! heap. Size from `CNV_CELLS` and `CNV_GENES` (default 10k x 10k). With
//! `CNV_DUMP=<dir>` the counts, genes and truth are also written there as
//! plain text, so other tools can run on the identical matrix.
//!
//! Simulation: 20 equal chromosomes, log-normal gene means, PLN noise
//! (sigma 0.3), depth uniform in [2000, 6000). A third of the cells are
//! normal, a third carry a gain on chr2, a third carry that gain plus a loss
//! on chr5.

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cnv_sc_rs::cnv::{CnvParams, run_cnv};
use cnv_sc_rs::genome::CsrCounts;
use cnv_sc_rs::prelude::Verbosity;
use cnv_sc_rs::reference::Reference;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal, Normal, Poisson};
use rayon::prelude::*;

/// Heap allocator that tracks live and peak bytes.
struct Counting;

/// Live heap bytes.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Peak of `LIVE` since the last reset.
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        PEAK.fetch_max(now, Ordering::Relaxed);
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Chromosomes, equal size.
const N_CHROM: usize = 20;

/// Read a size from the environment.
///
/// ### Params
///
/// * `key` - Variable name
/// * `default` - Value when unset
///
/// ### Returns
///
/// The parsed size.
fn env_size(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| v.parse().expect("size"))
        .unwrap_or(default)
}

fn main() {
    let c = env_size("CNV_CELLS", 10_000);
    let g = env_size("CNV_GENES", 10_000);
    let per = g / N_CHROM;
    let g = per * N_CHROM;

    let t = Instant::now();
    let mut rng = StdRng::seed_from_u64(11);
    let ln = LogNormal::new(0.0, 1.5).unwrap();
    let mut lambda: Vec<f64> = (0..g).map(|_| ln.sample(&mut rng)).collect();
    let tot: f64 = lambda.iter().sum();
    lambda.iter_mut().for_each(|l| *l /= tot);
    let chrom: Vec<u32> = (0..g).map(|i| 1 + (i / per) as u32).collect();
    let start: Vec<u64> = (0..g).map(|i| 1_000 + 10_000 * (i % per) as u64).collect();
    let end: Vec<u64> = start.iter().map(|s| s + 5_000).collect();
    let truth: Vec<u8> = (0..c).map(|_| rng.random_range(0..3u8)).collect();
    let phi = |clone: u8, i: usize| {
        let (ch, j) = (i / per, i % per);
        if clone >= 1 && ch == 1 && j < per * 2 / 5 {
            1.5
        } else if clone == 2 && ch == 4 && (per / 4..per * 3 / 4).contains(&j) {
            0.5
        } else {
            1.0
        }
    };
    let rows: Vec<Vec<(u32, u32)>> = truth
        .par_iter()
        .enumerate()
        .map(|(cell, &clone)| {
            let mut rng = StdRng::seed_from_u64(1_000 + cell as u64);
            let noise = Normal::<f64>::new(0.0, 0.3).unwrap();
            let d = rng.random_range(2_000.0..6_000.0);
            (0..g)
                .filter_map(|i| {
                    let rate = d * lambda[i] * phi(clone, i) * noise.sample(&mut rng).exp();
                    let y = Poisson::new(rate.max(1e-12)).unwrap().sample(&mut rng) as u32;
                    (y > 0).then_some((i as u32, y))
                })
                .collect()
        })
        .collect();
    let (mut indptr, mut indices, mut data) = (vec![0u32], Vec::new(), Vec::new());
    for r in &rows {
        indices.extend(r.iter().map(|x| x.0));
        data.extend(r.iter().map(|x| x.1));
        indptr.push(indices.len() as u32);
    }
    drop(rows);
    println!(
        "simulated {c} cells x {g} genes, nnz {} ({:.1}%), in {:.1}s",
        data.len(),
        100.0 * data.len() as f64 / (c * g) as f64,
        t.elapsed().as_secs_f64()
    );

    if let Ok(dir) = std::env::var("CNV_DUMP") {
        std::fs::create_dir_all(&dir).unwrap();
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(format!("{dir}/counts.mtx")).unwrap());
        writeln!(f, "%%MatrixMarket matrix coordinate integer general").unwrap();
        writeln!(f, "{c} {g} {}", data.len()).unwrap();
        for cell in 0..c {
            for k in indptr[cell] as usize..indptr[cell + 1] as usize {
                writeln!(f, "{} {} {}", cell + 1, indices[k] + 1, data[k]).unwrap();
            }
        }
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(format!("{dir}/genes.tsv")).unwrap());
        writeln!(f, "gene\tchromosome\tstart\tend\tlambda").unwrap();
        for i in 0..g {
            writeln!(
                f,
                "g{i}\tchr{}\t{}\t{}\t{:e}",
                chrom[i], start[i], end[i], lambda[i]
            )
            .unwrap();
        }
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(format!("{dir}/cells.tsv")).unwrap());
        writeln!(f, "cell\tclone").unwrap();
        for (i, t) in truth.iter().enumerate() {
            writeln!(f, "c{i}\t{t}").unwrap();
        }
        println!("dumped to {dir}");
    }

    let counts = CsrCounts::new(&indptr, &indices, &data, g).unwrap();
    let lambda32: Vec<f32> = lambda.iter().map(|&l| l as f32).collect();
    let reference = Reference::Profiles {
        lambda: &lambda32,
        n_ref: 1,
    };
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let t = Instant::now();
    let res = run_cnv::<f32>(
        &counts,
        &chrom,
        &start,
        &end,
        &reference,
        Some(CnvParams::default()),
        Verbosity::Normal,
    )
    .unwrap();
    let wall = t.elapsed().as_secs_f64();
    let peak = PEAK.load(Ordering::Relaxed) - base;
    println!(
        "total {wall:.2}s on {} threads, peak heap above input {:.1} MB, input {:.1} MB",
        rayon::current_num_threads(),
        peak as f64 / 1e6,
        (indices.len() * 8 + indptr.len() * 4) as f64 / 1e6
    );
    match res.clones.as_ref() {
        Some(cl) => {
            let ok = res
                .cells
                .iter()
                .enumerate()
                .filter(|&(row, &cell)| (cl.p_tumour[row] > 0.5) == (truth[cell as usize] > 0))
                .count();
            println!(
                "segments {}, tumour/normal accuracy {:.3}",
                res.consensus.len(),
                ok as f64 / res.cells.len() as f64
            );
        }
        None => println!("no CNV called"),
    }
}
