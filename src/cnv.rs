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
use crate::genome::{CsrCounts, GenomeOrder};
use crate::hmm::{BulkHmmInput, CnState, HmmParams, STATES, decode_bulk};
use crate::posterior::{
    CellPosteriors, ClonePosterior, GenotypeEvent, cell_posteriors, clone_posterior,
    effective_state,
};
use crate::reference::{
    Profiles, Reference, choose_ref_cor, filter_genes, fit_ref_sse, lambdas_bar,
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

////////////
// Params //
////////////

/// Parameters of [`run_cnv`]. Defaults are numbat's (N/R/main.R:68-77).
#[derive(Clone, Debug)]
pub struct CnvParams {
    /// Genes overlapping any `(chrom, start, end)` are excluded (MHC; SPEC 1)
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
}

impl Default for CnvParams {
    fn default() -> Self {
        Self {
            exclude: Vec::new(),
            smooth: SmoothParams::default(),
            init_k: 3,
            min_cells: 50,
            hmm: HmmParams::default(),
            min_genes: 10,
            consensus: ConsensusParams::default(),
            max_entropy: 0.5,
            p_multi: 1.0 - 1e-4,
            linkage: None,
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

/// Build a pseudobulk (SPEC 7.2): sum, fit the reference mixture, filter
/// genes, drop extreme `logFC` and `lambda = 0`.
///
/// ### Params
///
/// * `counts` - Count matrix
/// * `order` - Genomic order
/// * `profiles` - Reference profiles
/// * `allowed` - Exclusion mask, genomic order
/// * `cells` - The group's cells
///
/// ### Returns
///
/// The bulk, or an error (including a count beyond `u32`).
fn make_bulk(
    counts: &CsrCounts,
    order: &GenomeOrder,
    profiles: &Profiles,
    allowed: &[bool],
    cells: &[u32],
) -> Result<Bulk, CnvErrors> {
    let y_all = order.to_genomic(&counts.gene_sums(Some(cells))?)?;
    let fit = fit_ref_sse(&y_all, profiles)?;
    let d = y_all.iter().sum::<u64>() as f64;
    let keep = filter_genes(counts, order, &fit.lambda_bar, Some(cells), allowed)?;
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
/// ### Params
///
/// * `counts` - Cells x genes raw counts, caller gene order
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
    counts: &CsrCounts,
    chrom: &[u32],
    start: &[u64],
    end: &[u64],
    reference: &Reference<'_, T>,
    params: Option<CnvParams>,
    verbosity: Verbosity,
) -> Result<CnvResult<T>, CnvErrors> {
    let params = params.unwrap_or_default();
    let t = Instant::now();
    let order = GenomeOrder::new(chrom, start, end)?;
    let allowed = order.allowed_mask(&params.exclude);
    let profiles = reference.resolve(counts, &order)?;
    let refs = choose_ref_cor(counts, &order, &profiles)?;
    let cells: Vec<u32> = (0..counts.n_cells())
        .filter(|&c| refs[c].is_some() && counts.row(c).1.iter().any(|&v| v > 0))
        .map(|c| c as u32)
        .collect();
    if cells.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "cells with counts",
        });
    }
    let lambda_bar = lambdas_bar(&profiles, &refs)?;
    log_stage(verbosity, "reference", t);

    let t = Instant::now();
    let keep = filter_genes(counts, &order, &lambda_bar, Some(&cells), &allowed)?;
    let bins: SmoothedBins<T> =
        smooth_bins(counts, &order, &keep, &lambda_bar, Some(params.smooth))?;
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
    let bulks: Vec<Bulk> = groups
        .iter()
        .map(|g| make_bulk(counts, &order, &profiles, &allowed, &g.cells))
        .collect::<Result<_, _>>()?;
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
    let post = cell_posteriors(
        counts,
        &out.order,
        &profiles,
        &out.cells,
        &cell_refs,
        &out.consensus,
        ul,
    )?;
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
}
