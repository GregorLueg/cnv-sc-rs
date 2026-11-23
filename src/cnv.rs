//! The tier 0 pipeline, one pass (SPEC 10).
//!
//! Reference choice, smoothed bins, cell hierarchy, pseudobulk HMMs,
//! consensus, retest, per-cell posteriors and the clone posterior. Each stage
//! is a function of its own module; this file only wires them and owns the
//! pseudobulk construction (SPEC 7.2).

use std::time::Instant;

use ann_search_rs::Verbosity;
use bonsai_rs::tree::linkage::LinkageParams;

use crate::cluster::{Group, ward_groups};
use crate::errors::CnvErrors;
use crate::genome::{CellBlocks, EXCLUDE_HG38, GenomeOrder};
use crate::hmm::{BulkHmmInput, CnState, HmmParams, STATES, decode_bulk};
use crate::posterior::{
    CellPosteriors, ClonePosterior, GenotypeEvent, cell_posteriors, clone_posterior,
    effective_state, posterior_use_loh,
};
use crate::reference::{
    Profiles, Reference, choose_ref_cor, filter_genes_from_sums, fit_ref_sse, lambdas_bar,
    profile_from_sums, profiles_genomic,
};
use crate::segments::{
    BulkGenes, ConsensusParams, Genome, PostState, Segment, annot_segs, fit_baseline,
    retest_consensus, retest_segments, segs_consensus, smooth_segs, test_multi_allelic, use_loh,
};
use crate::smooth::{SmoothParams, SmoothedBins, smooth_bins};
use crate::utils::traits::CnvFloat;

/// Pseudobulk genes with `|logFC|` at or above this are dropped unless their
/// count is 0 (N/R/utils.R:368).
const BULK_MAX_ABS_LOGFC: f64 = 5.0;

/// Default cells per block read from a [`CellBlocks`] source. Our choice:
/// enough cells per block to keep every thread busy, small enough that one
/// block of counts is a minor share of memory.
pub const BLOCK_CELLS: usize = 8192;

////////////
// Params //
////////////

/// Parameters of [`run_cnv`]. Defaults are numbat's (N/R/main.R:68-77).
#[derive(Clone, Debug)]
pub struct CnvParams {
    /// Genes overlapping any `(chrom, start, end)` are excluded (SPEC 1);
    /// [`EXCLUDE_HG38`] by default, [`EXCLUDE_HG19`](crate::genome::EXCLUDE_HG19) for GRCh37
    pub exclude: Vec<(u32, u64, u64)>,
    /// Smoothing and binning (SPEC 3)
    pub smooth: SmoothParams,
    /// Leaf clusters of the k-cut, `init_k`
    pub init_k: usize,
    /// Groups need more than this many cells, `min_cells`
    pub min_cells: usize,
    /// HMM parameters (SPEC 5)
    pub hmm: HmmParams,
    /// Segments up to this many genes are smoothed away, `min_genes`
    pub min_genes: usize,
    /// Consensus parameters (SPEC 6.5)
    pub consensus: ConsensusParams,
    /// Segments with a higher mean `p_cnv` entropy are not confident, `max_entropy`
    pub max_entropy: f64,
    /// Probability threshold of the multi-state test, `p_multi`
    pub p_multi: f64,
    /// kNN-Ward knobs, `None` for bonsai's defaults
    pub linkage: Option<LinkageParams>,
    /// Cells per block read from the source, `>= 1`
    pub block_cells: usize,
}

impl Default for CnvParams {
    fn default() -> Self {
        Self {
            exclude: EXCLUDE_HG38.to_vec(),
            smooth: SmoothParams::default(),
            init_k: 3,
            min_cells: 50,
            hmm: HmmParams::default(),
            min_genes: 10,
            consensus: ConsensusParams::default(),
            max_entropy: 0.5,
            p_multi: 1.0 - 1e-4,
            linkage: None,
            block_cells: BLOCK_CELLS,
        }
    }
}

////////////
// Result //
////////////

/// Output of [`run_cnv`].
#[derive(Clone, Debug)]
pub struct CnvResult<T: CnvFloat> {
    /// Genomic order of the genes
    pub order: GenomeOrder,
    /// Reference profile per input cell, `None` for dropped cells
    pub refs: Vec<Option<u32>>,
    /// Cells used: non-zero total and a reference
    pub cells: Vec<u32>,
    /// Smoothed bins, one row per input cell
    pub bins: SmoothedBins<T>,
    /// Groups that passed `min_cells`; internal first, then leaves
    pub groups: Vec<Group>,
    /// Consensus segments after the retest on consensus (SPEC 6.6, 6.7)
    pub consensus: Vec<Segment>,
    /// Per consensus segment, its states if multi-state
    pub multi_state: Vec<Option<Vec<PostState>>>,
    /// Per-cell posteriors; `None` if no segment is non-neutral
    pub posteriors: Option<CellPosteriors>,
    /// Mean `p_cnv` entropy per posterior segment column
    pub entropy: Vec<f64>,
    /// Whether each posterior segment column is a confident CNV (SPEC 8.7)
    pub confident: Vec<bool>,
    /// Clone posterior; `None` if no segment is non-neutral
    pub clones: Option<ClonePosterior>,
}

////////////////
// Pseudobulk //
////////////////

/// One pseudobulk restricted to its genes (SPEC 7.2).
struct Bulk {
    /// Global genomic gene index per gene, ascending
    gene: Vec<u32>,
    /// Chromosome per gene
    chrom: Vec<u32>,
    /// Gene start per gene
    pos: Vec<u64>,
    /// Summed counts
    y: Vec<u32>,
    /// The bulk's `lambda_bar`
    lambda: Vec<f64>,
    /// Library size over all genes
    d: f64,
}

impl Bulk {
    /// Borrow as the segments' bulk view.
    ///
    /// ### Returns
    ///
    /// The view.
    fn genes(&self) -> BulkGenes<'_, f64> {
        BulkGenes {
            gene: &self.gene,
            y: &self.y,
            lambda: &self.lambda,
            d: self.d,
        }
    }
}

/// Build a pseudobulk (SPEC 7.2) from its gene sums: fit the reference
/// mixture, filter genes, drop extreme `logFC` and `lambda = 0`.
///
/// ### Params
///
/// * `sums` - The group's summed counts per gene, caller order
/// * `order` - Genomic order
/// * `profiles` - Reference profiles
/// * `allowed` - Exclusion mask, genomic order
///
/// ### Returns
///
/// The bulk, or an error (including a count beyond `u32`).
fn make_bulk(
    sums: &[u64],
    order: &GenomeOrder,
    profiles: &Profiles,
    allowed: &[bool],
) -> Result<Bulk, CnvErrors> {
    let y_all = order.to_genomic(sums)?;
    let fit = fit_ref_sse(&y_all, profiles)?;
    let d = y_all.iter().sum::<u64>() as f64;
    let keep = filter_genes_from_sums(&y_all, &fit.lambda_bar, allowed)?;
    let mut b = Bulk {
        gene: Vec::new(),
        chrom: Vec::new(),
        pos: Vec::new(),
        y: Vec::new(),
        lambda: Vec::new(),
        d,
    };
    for g in 0..order.n_genes() {
        let (y, lam) = (y_all[g], fit.lambda_bar[g]);
        if !keep[g] || lam <= 0.0 {
            continue;
        }
        let logfc = (y as f64 / d).log2() - lam.log2();
        if y != 0 && logfc.abs() >= BULK_MAX_ABS_LOGFC {
            continue;
        }
        let y = u32::try_from(y).map_err(|_| CnvErrors::CountOverflow { gene: g, count: y })?;
        b.gene.push(g as u32);
        b.chrom.push(order.chrom[g]);
        b.pos.push(order.start[g]);
        b.y.push(y);
        b.lambda.push(lam);
    }
    if b.gene.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "pseudobulk genes",
        });
    }
    Ok(b)
}

/// HMM, segment smoothing and retest on one bulk (SPEC 5, 6.1 to 6.4).
///
/// ### Params
///
/// * `bulk` - The bulk
/// * `id` - Bulk id stamped on the segments
/// * `params` - Pipeline parameters
///
/// ### Returns
///
/// Retested segments of the bulk.
fn call_bulk(bulk: &Bulk, id: u32, params: &CnvParams) -> Result<Vec<Segment>, CnvErrors> {
    let base = fit_baseline(&bulk.y, &bulk.lambda, bulk.d, &vec![true; bulk.y.len()])?;
    let path = decode_bulk(
        &BulkHmmInput {
            chrom: &bulk.chrom,
            y: &bulk.y,
            lambda: &bulk.lambda,
            d: bulk.d,
            mu: base.mu,
            sigma: base.sigma,
        },
        &params.hmm,
    )?;
    let states: Vec<CnState> = path.iter().map(|&s| STATES[s as usize].cn).collect();
    let states = smooth_segs(&bulk.chrom, &states, params.min_genes)?;
    let mut segs = annot_segs(Some(id), &bulk.chrom, &bulk.pos, &bulk.gene, &states)?;
    retest_segments(&bulk.genes(), &mut segs, &base, params.hmm.logphi_min)?;
    Ok(segs)
}

/// Block ranges covering `0..n`.
///
/// ### Params
///
/// * `n` - Number of cells
/// * `block` - Cells per block, `>= 1`
///
/// ### Returns
///
/// `(lo, hi)` per block, ascending.
fn blocks(n: usize, block: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n)
        .step_by(block)
        .map(move |lo| (lo, (lo + block).min(n)))
}

/// Members of an ascending cell list inside `lo..hi`, relative to `lo`.
///
/// ### Params
///
/// * `cells` - Ascending cell indices
/// * `lo` - First cell of the block
/// * `hi` - One past the last cell of the block
///
/// ### Returns
///
/// `(range in cells, block-relative indices)`.
fn in_block(cells: &[u32], lo: usize, hi: usize) -> (std::ops::Range<usize>, Vec<u32>) {
    let a = cells.partition_point(|&c| (c as usize) < lo);
    let b = cells.partition_point(|&c| (c as usize) < hi);
    (a..b, cells[a..b].iter().map(|&c| c - lo as u32).collect())
}

/// Add one vector of sums into another.
///
/// ### Params
///
/// * `acc` - Accumulator
/// * `x` - Sums to add, same length
fn add_sums(acc: &mut [u64], x: &[u64]) {
    acc.iter_mut().zip(x).for_each(|(a, b)| *a += b);
}

/// Print a stage timing line.
///
/// ### Params
///
/// * `v` - Verbosity
/// * `stage` - Stage name
/// * `t` - Stage start
fn log_stage(v: Verbosity, stage: &str, t: Instant) {
    if v.normal_verbosity() {
        println!(" {stage} in {:.2}s", t.elapsed().as_secs_f64());
    }
}

//////////////
// Pipeline //
//////////////

/// Expression-only CNV inference, one pass (SPEC 10).
///
/// Reads the cells four times, block by block (SPEC 10): reference choice
/// and gene sums, smoothing to bins, pseudobulk sums, per-cell posteriors.
/// Between passes it holds only per-gene sums, the bins and the outputs, so
/// counts never need to be in memory as a whole.
///
/// ### Params
///
/// * `counts` - Cells x genes raw counts, caller gene order, read in blocks
/// * `chrom` - Chromosome id per gene, caller order
/// * `start` - Gene start per gene
/// * `end` - Gene end per gene
/// * `reference` - Reference profiles or normal cells
/// * `params` - `None` for numbat's defaults
/// * `verbosity` - Stage timing lines at `Normal`
///
/// ### Returns
///
/// The result. If the consensus has no non-neutral segment, posteriors and
/// clones are `None`.
pub fn run_cnv<T: CnvFloat>(
    counts: &impl CellBlocks,
    chrom: &[u32],
    start: &[u64],
    end: &[u64],
    reference: &Reference<'_, T>,
    params: Option<CnvParams>,
    verbosity: Verbosity,
) -> Result<CnvResult<T>, CnvErrors> {
    let params = params.unwrap_or_default();
    if params.block_cells == 0 {
        return Err(CnvErrors::InvalidParameter {
            name: "block_cells",
            value: "0".into(),
            expected: ">= 1",
        });
    }
    let t = Instant::now();
    let order = GenomeOrder::new(chrom, start, end)?;
    let (n_cells, n_genes) = (counts.n_cells(), order.n_genes());
    if counts.n_genes() != n_genes {
        return Err(CnvErrors::ShapeMismatch {
            what: "genes in counts",
            expected: n_genes,
            got: counts.n_genes(),
        });
    }
    let allowed = order.allowed_mask(&params.exclude);
    let profiles = match *reference {
        Reference::Profiles { lambda, n_ref } => profiles_genomic(lambda, n_ref, &order)?,
        Reference::Cells(_) | Reference::Groups { .. } => {
            let lists = reference
                .cell_groups(n_cells)?
                .expect("cell references have groups");
            let mut sums = vec![vec![0u64; n_genes]; lists.len()];
            for (lo, hi) in blocks(n_cells, params.block_cells) {
                let rel: Vec<Vec<u32>> = lists.iter().map(|l| in_block(l, lo, hi).1).collect();
                if rel.iter().all(|r| r.is_empty()) {
                    continue;
                }
                counts.with_block(lo, hi, |b| {
                    for (acc, r) in sums.iter_mut().zip(&rel) {
                        if !r.is_empty() {
                            add_sums(acc, &b.gene_sums(Some(r))?);
                        }
                    }
                    Ok(())
                })?;
            }
            let mut lambda = Vec::with_capacity(lists.len() * n_genes);
            for s in &sums {
                lambda.extend(profile_from_sums(&order.to_genomic(s)?)?);
            }
            Profiles {
                n_ref: lists.len(),
                lambda,
            }
        }
    };

    // pass 1: reference per cell, kept cells, their gene sums
    let mut refs = Vec::with_capacity(n_cells);
    let mut cells = Vec::new();
    let mut sums = vec![0u64; n_genes];
    for (lo, hi) in blocks(n_cells, params.block_cells) {
        counts.with_block(lo, hi, |b| {
            let r = choose_ref_cor(b, &order, &profiles)?;
            let kept: Vec<u32> = (0..b.n_cells())
                .filter(|&c| r[c].is_some() && b.row(c).1.iter().any(|&v| v > 0))
                .map(|c| c as u32)
                .collect();
            add_sums(&mut sums, &b.gene_sums(Some(&kept))?);
            cells.extend(kept.iter().map(|&c| c + lo as u32));
            refs.extend(r);
            Ok(())
        })?;
    }
    if cells.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "cells with counts",
        });
    }
    let lambda_bar = lambdas_bar(&profiles, &refs)?;
    log_stage(verbosity, "reference", t);

    // pass 2: smoothing
    let t = Instant::now();
    let keep = filter_genes_from_sums(&order.to_genomic(&sums)?, &lambda_bar, &allowed)?;
    drop(sums);
    let mut bins: Option<SmoothedBins<T>> = None;
    for (lo, hi) in blocks(n_cells, params.block_cells) {
        let part: SmoothedBins<T> = counts.with_block(lo, hi, |b| {
            smooth_bins(b, &order, &keep, &lambda_bar, Some(params.smooth))
        })?;
        match bins.as_mut() {
            None => {
                let mut first = part;
                first.data.reserve_exact(first.bins.len() * (n_cells - hi));
                bins = Some(first);
            }
            Some(acc) => {
                acc.data.extend_from_slice(&part.data);
                acc.n_cells += part.n_cells;
            }
        }
    }
    let bins = bins.expect("n_cells > 0 since cells is non-empty");
    log_stage(verbosity, "smoothing", t);

    let t = Instant::now();
    let nb = bins.bins.len();
    let mut rows = Vec::with_capacity(cells.len() * nb);
    for &c in &cells {
        rows.extend_from_slice(&bins.data[c as usize * nb..(c as usize + 1) * nb]);
    }
    let groups: Vec<Group> = ward_groups(&rows, &cells, nb, params.init_k, params.linkage)?
        .into_iter()
        .filter(|g| g.cells.len() > params.min_cells)
        .collect();
    drop(rows);
    if groups.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "groups with more than min_cells cells",
        });
    }
    log_stage(verbosity, "clustering", t);

    let t = Instant::now();
    // pass 3: pseudobulk sums
    let mut group_sums = vec![vec![0u64; n_genes]; groups.len()];
    for (lo, hi) in blocks(n_cells, params.block_cells) {
        counts.with_block(lo, hi, |b| {
            for (g, acc) in groups.iter().zip(group_sums.iter_mut()) {
                let (_, rel) = in_block(&g.cells, lo, hi);
                if !rel.is_empty() {
                    add_sums(acc, &b.gene_sums(Some(&rel))?);
                }
            }
            Ok(())
        })?;
    }
    let bulks: Vec<Bulk> = group_sums
        .iter()
        .map(|y| make_bulk(y, &order, &profiles, &allowed))
        .collect::<Result<_, _>>()?;
    drop(group_sums);
    let mut segs = Vec::new();
    for (i, b) in bulks.iter().enumerate() {
        segs.extend(call_bulk(b, i as u32, &params)?);
    }
    log_stage(verbosity, "bulk HMMs", t);

    let t = Instant::now();
    let genome = Genome {
        chrom: &order.chrom,
        pos: &order.start,
    };
    let lp = params.hmm.logphi_min;
    let min_llr = params.consensus.min_llr;
    let first = segs_consensus(&segs, &genome, &params.consensus)?;
    let ul = use_loh(&first);
    let mut retested = Vec::new();
    for (i, b) in bulks.iter().enumerate() {
        retested.extend(retest_consensus(
            &b.genes(),
            i as u32,
            &first,
            ul,
            lp,
            min_llr,
        )?);
    }
    let mut consensus = segs_consensus(&retested, &genome, &params.consensus)?;
    let ul = use_loh(&consensus);
    let leaf_ids: Vec<usize> = (0..groups.len())
        .filter(|&i| groups[i].leaf.is_some())
        .collect();
    let leaves: Vec<Vec<Segment>> = leaf_ids
        .iter()
        .map(|&i| retest_consensus(&bulks[i].genes(), i as u32, &consensus, ul, lp, min_llr))
        .collect::<Result<_, _>>()?;
    let multi_state = test_multi_allelic(&mut consensus, &leaves, min_llr, params.p_multi)?;
    log_stage(verbosity, "consensus", t);

    let mut out = CnvResult {
        order,
        refs,
        cells,
        bins,
        groups,
        consensus,
        multi_state,
        posteriors: None,
        entropy: Vec::new(),
        confident: Vec::new(),
        clones: None,
    };
    if out
        .consensus
        .iter()
        .all(|s| effective_state(s) == PostState::Neu)
    {
        return Ok(out);
    }

    let t = Instant::now();
    let cell_refs: Vec<u32> = out
        .cells
        .iter()
        .map(|&c| out.refs[c as usize].expect("kept cells have a reference"))
        .collect();
    // pass 4: per-cell posteriors
    let post_ul = posterior_use_loh(&out.consensus);
    let mut post: Option<CellPosteriors> = None;
    for (lo, hi) in blocks(n_cells, params.block_cells) {
        let (range, rel) = in_block(&out.cells, lo, hi);
        if rel.is_empty() {
            continue;
        }
        let part = counts.with_block(lo, hi, |b| {
            cell_posteriors(
                b,
                &out.order,
                &profiles,
                &rel,
                &cell_refs[range.clone()],
                &out.consensus,
                post_ul,
            )
        })?;
        match post.as_mut() {
            None => {
                let mut first = part;
                first.cells = out.cells.clone();
                let row = first.log_odds.len() / rel.len();
                first
                    .log_odds
                    .reserve_exact(row * (out.cells.len() - rel.len()));
                post = Some(first);
            }
            Some(acc) => {
                acc.log_odds.extend_from_slice(&part.log_odds);
                acc.mu.extend_from_slice(&part.mu);
                acc.sigma.extend_from_slice(&part.sigma);
            }
        }
    }
    let post = post.expect("cells is non-empty");
    out.entropy = post.entropy();
    out.confident = post
        .segs
        .iter()
        .zip(&out.entropy)
        .map(|(&i, &h)| {
            h < params.max_entropy && out.consensus[i].retest.is_some_and(|r| r.llr > min_llr)
        })
        .collect();
    log_stage(verbosity, "cell posteriors", t);

    // genotypes: per leaf, its confident segments called non-neutral (SPEC 9)
    let genotypes: Vec<Vec<GenotypeEvent>> = leaves
        .iter()
        .map(|leaf| {
            post.segs
                .iter()
                .enumerate()
                .filter(|&(col, _)| out.confident[col])
                .filter_map(|(col, &i)| {
                    let st = effective_state(&leaf[i]);
                    (st != PostState::Neu).then(|| GenotypeEvent {
                        col,
                        state: out.multi_state[i].as_ref().map(|_| st),
                    })
                })
                .collect()
        })
        .collect();
    out.clones = Some(clone_posterior(&post, &genotypes));
    out.posteriors = Some(post);
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::CsrCounts;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, LogNormal, Normal, Poisson};

    /// 4 chromosomes x 750 genes. 200 normal cells, clone A (200) with a gain
    /// on chr 2 genes 0..400, clone B (200) with that gain plus a loss on
    /// chr 3 genes 200..600. Cell rows are shuffled. Returns CSR, chrom,
    /// start, end, reference profile and the truth per cell (0 normal, 1 A,
    /// 2 B).
    #[allow(clippy::type_complexity)]
    fn simulate(
        seed: u64,
    ) -> (
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u64>,
        Vec<u64>,
        Vec<f64>,
        Vec<u8>,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (n_chr, per) = (4u32, 750usize);
        let g = n_chr as usize * per;
        let ln = LogNormal::new(0.0, 1.5).unwrap();
        let mut lambda: Vec<f64> = (0..g).map(|_| ln.sample(&mut rng)).collect();
        let tot: f64 = lambda.iter().sum();
        lambda.iter_mut().for_each(|l| *l /= tot);
        let chrom: Vec<u32> = (0..g).map(|i| 1 + (i / per) as u32).collect();
        let start: Vec<u64> = (0..g).map(|i| 1_000 + 10_000 * (i % per) as u64).collect();
        let end: Vec<u64> = start.iter().map(|s| s + 5_000).collect();
        let phi = |clone: u8, i: usize| {
            let (c, j) = (i / per, i % per);
            if clone >= 1 && c == 1 && j < 400 {
                1.5
            } else if clone == 2 && c == 2 && (200..600).contains(&j) {
                0.5
            } else {
                1.0
            }
        };
        let mut truth: Vec<u8> = (0..600).map(|c| (c / 200) as u8).collect();
        // Fisher-Yates with the same rng
        for i in (1..truth.len()).rev() {
            let j = (rand::Rng::random::<u64>(&mut rng) % (i as u64 + 1)) as usize;
            truth.swap(i, j);
        }
        let noise = Normal::<f64>::new(0.0, 0.3).unwrap();
        let (mut indptr, mut indices, mut data) = (vec![0u32], Vec::new(), Vec::new());
        for &clone in &truth {
            let d = 4_000.0 * (0.5 + rand::Rng::random::<f64>(&mut rng));
            for i in 0..g {
                let rate = d * lambda[i] * phi(clone, i) * noise.sample(&mut rng).exp();
                let y = Poisson::new(rate.max(1e-12)).unwrap().sample(&mut rng) as u32;
                if y > 0 {
                    indices.push(i as u32);
                    data.push(y);
                }
            }
            indptr.push(indices.len() as u32);
        }
        (indptr, indices, data, chrom, start, end, lambda, truth)
    }

    #[test]
    fn test_run_cnv_recovers_planted_clones() {
        let (indptr, indices, data, chrom, start, end, lambda, truth) = simulate(7);
        let counts = CsrCounts::new(&indptr, &indices, &data, chrom.len()).unwrap();
        let reference = Reference::Profiles {
            lambda: &lambda,
            n_ref: 1,
        };
        let params = CnvParams {
            init_k: 3,
            ..Default::default()
        };
        let res = run_cnv::<f64>(
            &counts,
            &chrom,
            &start,
            &end,
            &reference,
            Some(params),
            Verbosity::Normal,
        )
        .unwrap();
        let calls: Vec<(u32, u32, u32, PostState)> = res
            .consensus
            .iter()
            .map(|s| (s.chrom, s.lo, s.hi, effective_state(s)))
            .filter(|c| c.3 != PostState::Neu)
            .collect();
        println!("calls {calls:?}");
        let clones = res.clones.as_ref().expect("CNVs called");
        println!("genotypes {:?}", clones.genotypes);
        // tumour/normal split
        let (mut tp, mut tn, mut n) = (0, 0, 0);
        for (row, &c) in res.cells.iter().enumerate() {
            let tumour = clones.p_tumour[row] > 0.5;
            let truth_t = truth[c as usize] > 0;
            n += 1;
            if tumour && truth_t {
                tp += 1;
            }
            if !tumour && !truth_t {
                tn += 1;
            }
        }
        let acc = (tp + tn) as f64 / n as f64;
        println!("tumour/normal accuracy {acc:.3}");
        // clone: the genotype's event count is the truth label here
        let hit = res
            .cells
            .iter()
            .enumerate()
            .filter(|&(row, &c)| {
                clones.genotypes[clones.clone[row] as usize].len() == truth[c as usize] as usize
            })
            .count();
        let clone_acc = hit as f64 / n as f64;
        println!("clone accuracy {clone_acc:.3}");
        assert!(clone_acc > 0.9);
        assert!(calls.iter().any(|c| c.0 == 2 && c.3 == PostState::Amp));
        assert!(calls.iter().any(|c| c.0 == 3 && c.3 == PostState::Del));
        assert!(acc > 0.95);
    }

    /// A source that copies each block into its own buffers, as a store would.
    struct Copying<'a>(CsrCounts<'a>);

    impl CellBlocks for Copying<'_> {
        fn n_cells(&self) -> usize {
            self.0.n_cells()
        }

        fn n_genes(&self) -> usize {
            self.0.n_genes
        }

        fn with_block<R>(
            &self,
            lo: usize,
            hi: usize,
            f: impl FnOnce(&CsrCounts<'_>) -> Result<R, CnvErrors>,
        ) -> Result<R, CnvErrors> {
            let (a, b) = (self.0.indptr[lo] as usize, self.0.indptr[hi] as usize);
            let indptr: Vec<u32> = self.0.indptr[lo..=hi]
                .iter()
                .map(|&p| p - a as u32)
                .collect();
            let (idx, val) = (self.0.indices[a..b].to_vec(), self.0.data[a..b].to_vec());
            f(&CsrCounts::new(&indptr, &idx, &val, self.0.n_genes)?)
        }
    }

    #[test]
    fn test_run_cnv_blocks_match_single_pass() {
        let (indptr, indices, data, chrom, start, end, lambda, truth) = simulate(7);
        // two empty cells first: with one-cell blocks, the leading blocks have
        // no kept cells
        let n_empty = 2u32;
        let indptr: Vec<u32> = std::iter::repeat_n(0, n_empty as usize)
            .chain(indptr)
            .collect();
        let counts = CsrCounts::new(&indptr, &indices, &data, chrom.len()).unwrap();
        let normal: Vec<u32> = (0..truth.len() as u32)
            .filter(|&c| truth[c as usize] == 0)
            .map(|c| c + n_empty)
            .rev()
            .collect();
        // a second profile with chr 4 shifted, renormalised
        let mut second: Vec<f64> = lambda
            .iter()
            .zip(&chrom)
            .map(|(&l, &c)| if c == 4 { 1.6 * l } else { l })
            .collect();
        let tot: f64 = second.iter().sum();
        second.iter_mut().for_each(|l| *l /= tot);
        let two: Vec<f64> = lambda.iter().chain(&second).copied().collect();
        let parity: Vec<u32> = normal.iter().map(|&c| c % 2).collect();
        let refs = [
            Reference::Profiles {
                lambda: &lambda,
                n_ref: 1,
            },
            Reference::Profiles {
                lambda: &two,
                n_ref: 2,
            },
            Reference::Cells(&normal),
            Reference::Groups {
                cells: &normal,
                group: &parity,
                n_ref: 2,
            },
        ];
        let copied = Copying(counts);
        let params = |block_cells| {
            Some(CnvParams {
                block_cells,
                ..Default::default()
            })
        };
        for (k, reference) in refs.iter().enumerate() {
            let whole = run_cnv(
                &counts,
                &chrom,
                &start,
                &end,
                reference,
                params(usize::MAX),
                Verbosity::Quiet,
            )
            .unwrap();
            assert!(whole.cells.iter().all(|&c| c >= n_empty));
            if k == 1 {
                // both profiles are chosen by some cells, so the per-cell
                // reference path is exercised
                for r in 0..2 {
                    assert!(whole.refs.contains(&Some(r)), "profile {r} unused");
                }
            }
            for block_cells in [37, 1] {
                let blocked = run_cnv(
                    &copied,
                    &chrom,
                    &start,
                    &end,
                    reference,
                    params(block_cells),
                    Verbosity::Quiet,
                )
                .unwrap();
                assert_eq!(whole.refs, blocked.refs);
                assert_eq!(whole.cells, blocked.cells);
                assert_eq!(whole.bins.data, blocked.bins.data);
                assert_eq!(whole.bins.n_cells, blocked.bins.n_cells);
                let segs = |r: &CnvResult<f64>| -> Vec<(u32, u32, u32)> {
                    r.consensus.iter().map(|s| (s.chrom, s.lo, s.hi)).collect()
                };
                assert_eq!(segs(&whole), segs(&blocked));
                let (pw, pb) = (
                    whole.posteriors.as_ref().unwrap(),
                    blocked.posteriors.as_ref().unwrap(),
                );
                assert_eq!(pw.cells, pb.cells);
                assert_eq!(pw.log_odds, pb.log_odds);
                assert_eq!(pw.mu, pb.mu);
                assert_eq!(
                    whole.clones.as_ref().unwrap().p_tumour,
                    blocked.clones.as_ref().unwrap().p_tumour
                );
            }
        }
    }

    #[test]
    fn test_run_cnv_groups_match_cells_and_profiles() {
        let (indptr, indices, data, chrom, start, end, _, truth) = simulate(7);
        let counts = CsrCounts::new(&indptr, &indices, &data, chrom.len()).unwrap();
        let normal: Vec<u32> = (0..truth.len() as u32)
            .filter(|&c| truth[c as usize] == 0)
            .collect();
        let run = |reference: &Reference<'_, f64>| {
            run_cnv(
                &counts,
                &chrom,
                &start,
                &end,
                reference,
                None,
                Verbosity::Quiet,
            )
            .unwrap()
        };
        let same = |a: &CnvResult<f64>, b: &CnvResult<f64>| {
            assert_eq!(a.refs, b.refs);
            assert_eq!(a.cells, b.cells);
            assert_eq!(a.bins.data, b.bins.data);
            assert_eq!(
                a.clones.as_ref().unwrap().p_tumour,
                b.clones.as_ref().unwrap().p_tumour
            );
        };

        // one label is the pooled reference
        let zeros = vec![0u32; normal.len()];
        let cells = run(&Reference::Cells(&normal));
        let one = run(&Reference::Groups {
            cells: &normal,
            group: &zeros,
            n_ref: 1,
        });
        same(&cells, &one);

        // two labels equal the profiles pooled from the same cells
        let parity: Vec<u32> = normal.iter().map(|&c| c % 2).collect();
        let groups = run(&Reference::Groups {
            cells: &normal,
            group: &parity,
            n_ref: 2,
        });
        let mut lambda = Vec::new();
        for r in 0..2 {
            let list: Vec<u32> = normal.iter().copied().filter(|&c| c % 2 == r).collect();
            lambda.extend(profile_from_sums(&counts.gene_sums(Some(&list)).unwrap()).unwrap());
        }
        let profiles = run(&Reference::Profiles {
            lambda: &lambda,
            n_ref: 2,
        });
        same(&groups, &profiles);
        for r in 0..2 {
            assert!(groups.refs.contains(&Some(r)), "profile {r} unused");
        }
    }
}
