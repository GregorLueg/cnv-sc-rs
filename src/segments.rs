//! Segments: annotation, smoothing, retest and consensus (SPEC 6).
//!
//! Genes are addressed by their global index in genomic order, so a segment's
//! gene range `[lo, hi)` means the same genes in every bulk. Base-pair
//! coordinates are closed intervals, as in numbat's `GRanges`. Consensus
//! steps that numbat runs in base pairs (neutral union, gaps) run in base
//! pairs here too; gene ranges of their output are derived from the genome.

use rayon::prelude::*;

use crate::errors::CnvErrors;
use crate::genome::chrom_runs;
use crate::hmm::CnState;
use crate::model::pln::{LnpoisFit, approx_phi_post, fit_lnpois, l_lnpois_multi};
use crate::utils::lse::{log_sum_exp, log1mexp};
use crate::utils::traits::{CnvFloat, wide};

///////////////
// Constants //
///////////////

/// Lower end of the deletion `phi` range in the retest (N/R/utils.R:757).
const PHI_RANGE_LO: f64 = 0.1;
/// Upper end of the gain `phi` range in the retest (N/R/utils.R:758).
const PHI_RANGE_HI: f64 = 3.0;

/// Retest prior on `neu`, `G['11']` (N/R/utils.R:702).
const PRIOR_NEU: f64 = 0.5;
/// Retest prior on `loh`, `G['20']` (N/R/utils.R:702).
const PRIOR_LOH: f64 = 0.1;
/// Retest prior on `del`, `G['10']` (N/R/utils.R:702).
const PRIOR_DEL: f64 = 0.1;
/// Retest prior on each of the two gains, `G['21']` and `G['31']`
/// (N/R/utils.R:702).
const PRIOR_AMP_EACH: f64 = 0.05;
/// Retest prior on `bamp`, `G['22']` (N/R/utils.R:702).
const PRIOR_BAMP: f64 = 0.1;
/// Retest prior on `bdel`, `G['00']` (N/R/utils.R:702).
const PRIOR_BDEL: f64 = 0.1;

/// Baseline genes need `|logFC| < 8` (N/R/utils.R:541).
const BASELINE_MAX_ABS_LOGFC: f64 = 8.0;

/// `use_loh` is on if the neutral consensus spans fewer bp (N/R/main.R:1631).
const USE_LOH_MAX_NEU_BP: f64 = 1.5e8;

/// Prior a multi-state segment gives each present state (N/R/main.R:1718).
const MULTI_STATE_PRIOR: f64 = 0.5;

/// Cut-over from `erfc` to the asymptotic series in [`ln_pnorm`]. At
/// `x = -30` the first omitted term is about `2e-14` relative.
const LN_PNORM_ASYMPTOTIC_BELOW: f64 = -30.0;

///////////
// Types //
///////////

/// Retest state (`cnv_state_post`). Declared in alphabetical order so that
/// sorting matches numbat's `sort(unique(.))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PostState {
    /// Gain
    Amp,
    /// Balanced gain
    Bamp,
    /// Balanced deletion
    Bdel,
    /// Deletion
    Del,
    /// Copy-neutral LOH
    Loh,
    /// Neutral
    Neu,
}

/// Result of the expression-only retest of one segment (SPEC 6.4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Retest {
    /// Laplace posterior mode of `phi`
    pub phi_hat: f64,
    /// Laplace posterior sd of `phi`; 0 when the fit failed
    pub s_phi: f64,
    /// `Z_cnv - Z_n`
    pub llr: f64,
    /// `l(phi_hat) - l(1)`
    pub llr_x: f64,
    /// Posterior probability of neutral
    pub p_neu: f64,
    /// Probability of LOH, conditional on a CNV
    pub p_loh: f64,
    /// Probability of a gain, conditional on a CNV (`bamp` folded in, D5)
    pub p_amp: f64,
    /// Probability of a deletion, conditional on a CNV (`bdel` folded in, D5)
    pub p_del: f64,
    /// Probability of a balanced gain, conditional on a CNV (0 after D5)
    pub p_bamp: f64,
    /// Probability of a balanced deletion, conditional on a CNV (0 after D5)
    pub p_bdel: f64,
    /// Retest state, after D5 one of `neu`, `amp`, `del`
    pub state_post: PostState,
}

/// One segment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    /// Bulk the segment comes from; `None` for neutral fill
    pub bulk: Option<u32>,
    /// Chromosome id
    pub chrom: u32,
    /// Ordinal within the chromosome, from 1 (numbat's letter postfix)
    pub seg: u32,
    /// First global gene index
    pub lo: u32,
    /// One past the last global gene index
    pub hi: u32,
    /// Start in bp, closed
    pub start: u64,
    /// End in bp, closed
    pub end: u64,
    /// Number of genes the segment was called on
    pub n_genes: u32,
    /// Copy number label from the HMM
    pub state: CnState,
    /// Retest, `None` for untested (neutral) segments
    pub retest: Option<Retest>,
}

/// A pseudobulk for the retest. Genes in genomic order.
#[derive(Clone, Copy, Debug)]
pub struct BulkGenes<'a, T: CnvFloat> {
    /// Global gene index per gene, strictly increasing
    pub gene: &'a [u32],
    /// Counts
    pub y: &'a [u32],
    /// Reference proportions (`lambda_bar`), all `> 0`
    pub lambda: &'a [T],
    /// Library size over all annotated genes
    pub d: f64,
}

/// The whole gene annotation in genomic order. Gene `i` has global index `i`.
#[derive(Clone, Copy, Debug)]
pub struct Genome<'a> {
    /// Chromosome per gene, contiguous runs
    pub chrom: &'a [u32],
    /// Position (gene start) per gene, non-decreasing within a chromosome
    pub pos: &'a [u64],
}

/// Consensus parameters (SPEC 6.5).
#[derive(Clone, Copy, Debug)]
pub struct ConsensusParams {
    /// Segments with a lower `LLR` count as neutral
    pub min_llr: f64,
    /// Overlap fraction that links two segments
    pub min_overlap: f64,
}

impl ConsensusParams {
    /// Create parameters.
    ///
    /// ### Params
    ///
    /// * `min_llr` - `LLR` threshold
    /// * `min_overlap` - Overlap fraction, in `[0, 1]`
    ///
    /// ### Returns
    ///
    /// The parameter struct.
    pub fn new(min_llr: f64, min_overlap: f64) -> Self {
        Self {
            min_llr,
            min_overlap,
        }
    }
}

impl Default for ConsensusParams {
    /// numbat defaults, N/R/main.R:68-77.
    fn default() -> Self {
        Self {
            min_llr: 5.0,
            min_overlap: 0.45,
        }
    }
}

////////////////////
// Annot / smooth //
////////////////////

/// Check that per-gene inputs agree in length.
///
/// ### Params
///
/// * `n` - Expected length
/// * `lens` - `(name, length)` pairs
///
/// ### Returns
///
/// `Ok(())` or a `ShapeMismatch`.
fn check_lens(n: usize, lens: &[(&'static str, usize)]) -> Result<(), CnvErrors> {
    for &(what, got) in lens {
        if got != n {
            return Err(CnvErrors::ShapeMismatch {
                what,
                expected: n,
                got,
            });
        }
    }
    Ok(())
}

/// Run-length segments of the copy number label per chromosome
/// (SPEC 6.2, `annot_segs`).
///
/// ### Params
///
/// * `bulk` - Bulk id stamped on the segments
/// * `chrom` - Chromosome per gene, contiguous runs
/// * `pos` - Position per gene
/// * `gene` - Global gene index per gene, strictly increasing
/// * `states` - Copy number label per gene
///
/// ### Returns
///
/// Segments in genomic order, untested, or an input error.
pub fn annot_segs(
    bulk: Option<u32>,
    chrom: &[u32],
    pos: &[u64],
    gene: &[u32],
    states: &[CnState],
) -> Result<Vec<Segment>, CnvErrors> {
    check_lens(
        chrom.len(),
        &[
            ("pos", pos.len()),
            ("gene", gene.len()),
            ("states", states.len()),
        ],
    )?;
    if gene.windows(2).any(|w| w[0] >= w[1]) {
        return Err(CnvErrors::NotGenomicOrder {
            what: "gene indices are not strictly increasing",
        });
    }
    let mut out = Vec::new();
    for (clo, chi) in chrom_runs(chrom)? {
        let mut seg = 0;
        let mut lo = clo;
        for i in clo + 1..=chi {
            if i == chi || states[i] != states[lo] {
                seg += 1;
                let p = &pos[lo..i];
                out.push(Segment {
                    bulk,
                    chrom: chrom[lo],
                    seg,
                    lo: gene[lo],
                    hi: gene[i - 1] + 1,
                    start: *p.iter().min().expect("non-empty run"),
                    end: *p.iter().max().expect("non-empty run"),
                    n_genes: (i - lo) as u32,
                    state: states[lo],
                    retest: None,
                });
                lo = i;
            }
        }
    }
    Ok(out)
}

/// Drop short segments and fill their genes from the neighbours
/// (SPEC 6.3, `smooth_segs`).
///
/// Runs of `<= min_genes` genes lose their label; labels are then filled
/// forward, then backward, within the chromosome. **D3:** a chromosome with
/// no run longer than `min_genes` becomes `neu` (numbat errors).
///
/// ### Params
///
/// * `chrom` - Chromosome per gene, contiguous runs
/// * `states` - Copy number label per gene
/// * `min_genes` - Runs up to this length are dropped
///
/// ### Returns
///
/// Smoothed labels per gene; call [`annot_segs`] on them again.
pub fn smooth_segs(
    chrom: &[u32],
    states: &[CnState],
    min_genes: usize,
) -> Result<Vec<CnState>, CnvErrors> {
    check_lens(chrom.len(), &[("states", states.len())])?;
    let mut kept: Vec<Option<CnState>> = Vec::with_capacity(states.len());
    let runs = chrom_runs(chrom)?;
    for &(clo, chi) in &runs {
        let mut lo = clo;
        for i in clo + 1..=chi {
            if i == chi || states[i] != states[lo] {
                let s = if i - lo <= min_genes {
                    None
                } else {
                    Some(states[lo])
                };
                kept.extend(std::iter::repeat_n(s, i - lo));
                lo = i;
            }
        }
    }
    let mut out = Vec::with_capacity(states.len());
    for (clo, chi) in runs {
        let part = &mut kept[clo..chi];
        let mut last = None;
        for s in part.iter_mut() {
            if s.is_some() {
                last = *s;
            } else {
                *s = last;
            }
        }
        let mut next = None;
        for s in part.iter_mut().rev() {
            if s.is_some() {
                next = *s;
            } else {
                *s = next;
            }
        }
        out.extend(part.iter().map(|s| s.unwrap_or(CnState::Neu)));
    }
    Ok(out)
}

////////////
// Retest //
////////////

/// `ln Phi(x)`, the standard normal log-CDF, accurate in both tails.
///
/// ### Params
///
/// * `x` - Argument
///
/// ### Returns
///
/// `ln Phi(x)`.
fn ln_pnorm(x: f64) -> f64 {
    use std::f64::consts::{FRAC_1_SQRT_2, PI};
    if x > 0.0 {
        (-0.5 * libm::erfc(x * FRAC_1_SQRT_2)).ln_1p()
    } else if x > LN_PNORM_ASYMPTOTIC_BELOW {
        (0.5 * libm::erfc(-x * FRAC_1_SQRT_2)).ln()
    } else {
        // Phi(x) = phi(x)/|x| * (1 - 1/x^2 + 3/x^4 - 15/x^6 + 105/x^8 - 945/x^10 ...)
        let r = 1.0 / (x * x);
        let s = 1.0 - r * (1.0 - r * (3.0 - r * (15.0 - r * (105.0 - r * 945.0))));
        -0.5 * x * x - (-x).ln() - 0.5 * (2.0 * PI).ln() + s.ln()
    }
}

/// Log of a normal probability mass on `[lower, upper]`
/// (`pnorm.range.log`, N/R/utils.R:1814-1819).
///
/// ### Params
///
/// * `lower` - Lower bound
/// * `upper` - Upper bound
/// * `mu` - Mean
/// * `sd` - Standard deviation
///
/// ### Returns
///
/// The log mass; **1** when `sd == 0`, as numbat.
fn pnorm_range_log(lower: f64, upper: f64, mu: f64, sd: f64) -> f64 {
    if sd == 0.0 {
        return 1.0;
    }
    let lu = ln_pnorm((upper - mu) / sd);
    let ll = ln_pnorm((lower - mu) / sd);
    lu + log1mexp(lu - ll)
}

/// numbat's retest probabilities from the `phi` posterior, alleles absent,
/// before D5 (SPEC 6.4, N/R/utils.R:740-786).
///
/// ### Params
///
/// * `phi_hat` - Posterior mode of `phi`
/// * `s_phi` - Posterior sd of `phi`
/// * `logphi_min` - `log2` fold change of a single-copy event
///
/// ### Returns
///
/// [`Retest`] with `llr_x = 0` and numbat's raw state and probabilities.
fn retest_raw(phi_hat: f64, s_phi: f64, logphi_min: f64) -> Retest {
    let (f_lo, f_hi) = ((-logphi_min).exp2(), logphi_min.exp2());
    let l_n = pnorm_range_log(f_lo, f_hi, phi_hat, s_phi);
    let l_d = pnorm_range_log(PHI_RANGE_LO, f_lo, phi_hat, s_phi);
    let l_a = pnorm_range_log(f_hi, PHI_RANGE_HI, phi_hat, s_phi);
    let z_cnv = log_sum_exp(&[
        PRIOR_LOH.ln() + l_n,
        PRIOR_DEL.ln() + l_d,
        PRIOR_AMP_EACH.ln() + l_a,
        PRIOR_AMP_EACH.ln() + l_a,
        PRIOR_BAMP.ln() + l_a,
        PRIOR_BDEL.ln() + l_d,
    ]);
    let z_n = PRIOR_NEU.ln() + l_n;
    let z = log_sum_exp(&[z_n, z_cnv]);
    // numbat sets NA probabilities to 0
    let p = |v: f64| {
        let e = v.exp();
        if e.is_nan() { 0.0 } else { e }
    };
    let p_neu = p(z_n - z);
    let cand = [
        (PostState::Loh, p(PRIOR_LOH.ln() + l_n - z_cnv)),
        (PostState::Amp, p((2.0 * PRIOR_AMP_EACH).ln() + l_a - z_cnv)),
        (PostState::Del, p(PRIOR_DEL.ln() + l_d - z_cnv)),
        (PostState::Bamp, p(PRIOR_BAMP.ln() + l_a - z_cnv)),
        (PostState::Bdel, p(PRIOR_BDEL.ln() + l_d - z_cnv)),
    ];
    let mut best = 0;
    for k in 1..cand.len() {
        if cand[k].1 > cand[best].1 {
            best = k;
        }
    }
    Retest {
        phi_hat,
        s_phi,
        llr: z_cnv - z_n,
        llr_x: 0.0,
        p_neu,
        p_loh: cand[0].1,
        p_amp: cand[1].1,
        p_del: cand[2].1,
        p_bamp: cand[3].1,
        p_bdel: cand[4].1,
        state_post: if p_neu >= 0.5 {
            PostState::Neu
        } else {
            cand[best].0
        },
    }
}

/// Apply D5: `loh -> neu`, `bamp -> amp`, `bdel -> del`, after the argmax.
/// `p_bamp` folds into `p_amp` and `p_bdel` into `p_del`; `p_loh` is kept.
///
/// ### Params
///
/// * `r` - Raw retest
///
/// ### Returns
///
/// The folded retest.
fn fold_d5(mut r: Retest) -> Retest {
    r.state_post = match r.state_post {
        PostState::Loh => PostState::Neu,
        PostState::Bamp => PostState::Amp,
        PostState::Bdel => PostState::Del,
        s => s,
    };
    r.p_amp += r.p_bamp;
    r.p_del += r.p_bdel;
    r.p_bamp = 0.0;
    r.p_bdel = 0.0;
    r
}

/// Expression-only retest of one segment (SPEC 6.4, D4, D5).
///
/// ### Params
///
/// * `y` - Counts of the segment's genes
/// * `lambda` - Reference proportions, all `> 0`
/// * `d` - Library size
/// * `mu` - Baseline `mu`
/// * `sigma` - Baseline `sigma`
/// * `logphi_min` - `log2` fold change of a single-copy event
///
/// ### Returns
///
/// The [`Retest`], or an input error.
pub fn retest_segment<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    mu: f64,
    sigma: f64,
    logphi_min: f64,
) -> Result<Retest, CnvErrors> {
    let (phi_hat, s_phi) = approx_phi_post(y, lambda, d, mu, sigma)?;
    let llr_x = if y.is_empty() {
        0.0
    } else {
        let l = l_lnpois_multi(y, lambda, d, mu, sigma, &[phi_hat, 1.0])?;
        l[0] - l[1]
    };
    let mut r = fold_d5(retest_raw(phi_hat, s_phi, logphi_min));
    r.llr_x = llr_x;
    Ok(r)
}

/// Bulk baseline fit (SPEC 6.1).
///
/// `fit_lnpois` on diploid genes with finite `logFC = log2(y/d) - log2
/// lambda` in `(-8, 8)`; if none remain, on all genes.
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, all `> 0`
/// * `d` - Library size
/// * `diploid` - Per gene, whether it counts as diploid
///
/// ### Returns
///
/// The fit, or an input or convergence error.
pub fn fit_baseline<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    diploid: &[bool],
) -> Result<LnpoisFit, CnvErrors> {
    check_lens(
        y.len(),
        &[("lambda", lambda.len()), ("diploid", diploid.len())],
    )?;
    let keep: Vec<usize> = (0..y.len())
        .filter(|&g| {
            let lfc = (f64::from(y[g]) / d).log2() - wide(lambda[g]).log2();
            diploid[g] && lfc.is_finite() && lfc.abs() < BASELINE_MAX_ABS_LOGFC
        })
        .collect();
    if keep.is_empty() {
        return fit_lnpois(y, lambda, d);
    }
    let ys: Vec<u32> = keep.iter().map(|&g| y[g]).collect();
    let ls: Vec<T> = keep.iter().map(|&g| lambda[g]).collect();
    fit_lnpois(&ys, &ls, d)
}

/// Map each gene of a bulk to the first segment whose gene range holds it.
///
/// ### Params
///
/// * `gene` - Global gene index per bulk gene, strictly increasing
/// * `segs` - Segments, in priority order
///
/// ### Returns
///
/// Segment index per gene, `None` where no segment covers it.
fn assign_genes(gene: &[u32], segs: &[Segment]) -> Vec<Option<usize>> {
    let mut out = vec![None; gene.len()];
    for (s, seg) in segs.iter().enumerate() {
        let a = gene.partition_point(|&g| g < seg.lo);
        let b = gene.partition_point(|&g| g < seg.hi);
        for o in &mut out[a..b] {
            o.get_or_insert(s);
        }
    }
    out
}

/// Validate a [`BulkGenes`].
///
/// ### Params
///
/// * `bulk` - The bulk
///
/// ### Returns
///
/// `Ok(())` or an input error.
fn check_bulk<T: CnvFloat>(bulk: &BulkGenes<T>) -> Result<(), CnvErrors> {
    check_lens(
        bulk.gene.len(),
        &[("y", bulk.y.len()), ("lambda", bulk.lambda.len())],
    )?;
    if bulk.gene.windows(2).any(|w| w[0] >= w[1]) {
        return Err(CnvErrors::NotGenomicOrder {
            what: "gene indices are not strictly increasing",
        });
    }
    Ok(())
}

/// Retest every non-neutral segment of a bulk in parallel (SPEC 6.4, D4).
///
/// Each segment's genes are the bulk genes in its gene range.
///
/// ### Params
///
/// * `bulk` - The bulk the segments were called on
/// * `segs` - Its segments; non-`neu` ones get `retest` filled
/// * `baseline` - Bulk baseline (SPEC 6.1)
/// * `logphi_min` - `log2` fold change of a single-copy event
///
/// ### Returns
///
/// `Ok(())`, or the first input error.
pub fn retest_segments<T: CnvFloat>(
    bulk: &BulkGenes<T>,
    segs: &mut [Segment],
    baseline: &LnpoisFit,
    logphi_min: f64,
) -> Result<(), CnvErrors> {
    check_bulk(bulk)?;
    let owner = assign_genes(bulk.gene, segs);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); segs.len()];
    for (g, o) in owner.iter().enumerate() {
        if let Some(s) = o {
            members[*s].push(g);
        }
    }
    segs.par_iter_mut().zip(&members).try_for_each(|(seg, m)| {
        if seg.state == CnState::Neu {
            return Ok(());
        }
        let y: Vec<u32> = m.iter().map(|&g| bulk.y[g]).collect();
        let l: Vec<T> = m.iter().map(|&g| bulk.lambda[g]).collect();
        seg.retest = Some(retest_segment(
            &y,
            &l,
            bulk.d,
            baseline.mu,
            baseline.sigma,
            logphi_min,
        )?);
        Ok(())
    })
}

///////////////
// Consensus //
///////////////

/// Minimal union-find over `0..n`.
struct UnionFind {
    /// Parent per node
    parent: Vec<usize>,
}

impl UnionFind {
    /// `n` singletons.
    ///
    /// ### Params
    ///
    /// * `n` - Number of nodes
    ///
    /// ### Returns
    ///
    /// The structure.
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    /// Root of `x`, with path halving.
    ///
    /// ### Params
    ///
    /// * `x` - Node
    ///
    /// ### Returns
    ///
    /// Its root.
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    /// Merge the sets of `a` and `b`.
    ///
    /// ### Params
    ///
    /// * `a` - Node
    /// * `b` - Node
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra.max(rb)] = ra.min(rb);
        }
    }
}

/// `LLR` used for consensus: `NaN` when untested.
///
/// ### Params
///
/// * `s` - Segment
///
/// ### Returns
///
/// `(LLR, LLR_x)`.
fn llrs(s: &Segment) -> (f64, f64) {
    s.retest.map_or((f64::NAN, f64::NAN), |r| (r.llr, r.llr_x))
}

/// Unique CNVs across bulks (SPEC 6.5 step 2, `resolve_cnvs`,
/// N/R/main.R:1116-1175).
///
/// Two segments on one chromosome are linked when their closed gene index
/// intervals `[lo, hi - 1]` overlap and the overlap covers at least
/// `min_overlap` of either. In each connected component, all segments of the
/// bulk with the highest per-bulk max `LLR_x` are kept; ties go to the bulk
/// listed first.
///
/// ### Params
///
/// * `segs` - Non-neutral segments of all bulks
/// * `min_overlap` - Overlap fraction
///
/// ### Returns
///
/// The kept segments, ordered by chromosome then start.
pub fn resolve_cnvs(segs: &[Segment], min_overlap: f64) -> Vec<Segment> {
    let n = segs.len();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in i + 1..n {
            let (x, y) = (&segs[i], &segs[j]);
            if x.chrom != y.chrom {
                continue;
            }
            let (sx, ex) = (x.lo as f64, x.hi as f64 - 1.0);
            let (sy, ey) = (y.lo as f64, y.hi as f64 - 1.0);
            if sx.max(sy) > ex.min(ey) {
                continue;
            }
            let ov = ex.min(ey) - sx.max(sy);
            // 0/0 is NA in numbat and its edge survives only if the other
            // fraction passes; counting NaN as short reproduces that
            let short = |f: f64| f.is_nan() || f < min_overlap;
            let cut = short(ov / (ex - sx)) && short(ov / (ey - sy));
            if !cut {
                uf.union(i, j);
            }
        }
    }
    // component ids in order of first vertex, as igraph::components
    let mut comp_of_root = vec![usize::MAX; n];
    let mut comp = vec![0; n];
    let mut n_comp = 0;
    for i in 0..n {
        let r = uf.find(i);
        if comp_of_root[r] == usize::MAX {
            comp_of_root[r] = n_comp;
            n_comp += 1;
        }
        comp[i] = comp_of_root[r];
    }
    // per (component, bulk) max LLR_x; winner per component in input order
    let mut bulk_max: Vec<Vec<(Option<u32>, f64)>> = vec![Vec::new(); n_comp];
    for i in 0..n {
        let v = llrs(&segs[i]).1;
        let e = &mut bulk_max[comp[i]];
        match e.iter_mut().find(|(b, _)| *b == segs[i].bulk) {
            Some((_, m)) => *m = m.max(v),
            None => e.push((segs[i].bulk, v)),
        }
    }
    let lookup = |c: usize, b: Option<u32>| {
        bulk_max[c]
            .iter()
            .find(|(x, _)| *x == b)
            .map_or(f64::NAN, |&(_, m)| m)
    };
    let mut winner: Vec<Option<(Option<u32>, f64)>> = vec![None; n_comp];
    for i in 0..n {
        let v = lookup(comp[i], segs[i].bulk);
        let w = &mut winner[comp[i]];
        if w.is_none_or(|(_, best)| v > best) && !v.is_nan() {
            *w = Some((segs[i].bulk, v));
        }
    }
    let mut kept: Vec<usize> = (0..n)
        .filter(|&i| winner[comp[i]].is_some_and(|(b, _)| b == segs[i].bulk))
        .collect();
    kept.sort_by(|&a, &b| {
        let (sa, sb) = (&segs[a], &segs[b]);
        (sa.chrom, sa.start, comp[a])
            .cmp(&(sb.chrom, sb.start, comp[b]))
            .then(a.cmp(&b))
    });
    kept.into_iter().map(|i| segs[i]).collect()
}

/// Closed bp intervals per chromosome, sorted by `(chrom, start)`.
type Ranges = Vec<(u32, u64, u64)>;

/// Union of closed intervals, merging overlapping and adjacent ones
/// (`GenomicRanges::reduce`).
///
/// ### Params
///
/// * `iv` - Intervals `(chrom, start, end)`
///
/// ### Returns
///
/// Disjoint sorted intervals.
fn reduce_ranges(mut iv: Ranges) -> Ranges {
    iv.sort_unstable();
    let mut out: Ranges = Vec::with_capacity(iv.len());
    for (c, s, e) in iv {
        match out.last_mut() {
            Some((lc, _, le)) if *lc == c && s <= le.saturating_add(1) => *le = (*le).max(e),
            _ => out.push((c, s, e)),
        }
    }
    out
}

/// `x` minus the union of `y`, closed intervals (`GenomicRanges::setdiff`).
///
/// ### Params
///
/// * `x` - Intervals
/// * `y` - Intervals to remove
///
/// ### Returns
///
/// Disjoint sorted intervals.
fn setdiff_ranges(x: Ranges, y: Ranges) -> Ranges {
    let x = reduce_ranges(x);
    let y = reduce_ranges(y);
    let mut out = Vec::new();
    for (c, s, e) in x {
        let mut cur = s;
        for &(_, ys, ye) in y.iter().filter(|r| r.0 == c && r.2 >= s && r.1 <= e) {
            if ys > cur {
                out.push((c, cur, ys - 1));
            }
            cur = cur.max(ye.saturating_add(1));
        }
        if cur <= e {
            out.push((c, cur, e));
        }
    }
    out
}

/// Validate a [`Genome`] and return its chromosome ranges.
///
/// ### Params
///
/// * `genome` - Gene annotation
///
/// ### Returns
///
/// `(chrom, lo, hi)` per chromosome, or an input error.
fn genome_runs(genome: &Genome) -> Result<Vec<(u32, usize, usize)>, CnvErrors> {
    check_lens(genome.chrom.len(), &[("pos", genome.pos.len())])?;
    let runs = chrom_runs(genome.chrom)?;
    for &(lo, hi) in &runs {
        if genome.pos[lo..hi].windows(2).any(|w| w[0] > w[1]) {
            return Err(CnvErrors::NotGenomicOrder {
                what: "positions decrease within a chromosome",
            });
        }
    }
    Ok(runs
        .into_iter()
        .map(|(lo, hi)| (genome.chrom[lo], lo, hi))
        .collect())
}

/// Global gene range `[lo, hi)` of a closed bp interval.
///
/// ### Params
///
/// * `genome` - Gene annotation
/// * `runs` - From [`genome_runs`]
/// * `chrom` - Chromosome
/// * `start` - Start, closed
/// * `end` - End, closed
///
/// ### Returns
///
/// Genes with `start <= pos <= end`; empty at 0 for an unknown chromosome.
fn gene_range(
    genome: &Genome,
    runs: &[(u32, usize, usize)],
    chrom: u32,
    start: u64,
    end: u64,
) -> (u32, u32) {
    let Some(&(_, clo, chi)) = runs.iter().find(|r| r.0 == chrom) else {
        return (0, 0);
    };
    let p = &genome.pos[clo..chi];
    let lo = clo + p.partition_point(|&x| x < start);
    let hi = clo + p.partition_point(|&x| x <= end);
    (lo as u32, hi.max(lo) as u32)
}

/// Neutral segment over a bp interval, gene range from the genome.
///
/// ### Params
///
/// * `genome` - Gene annotation
/// * `runs` - From [`genome_runs`]
/// * `iv` - `(chrom, start, end)`
///
/// ### Returns
///
/// An untested `neu` segment with no bulk; `seg` is set later.
fn neu_segment(genome: &Genome, runs: &[(u32, usize, usize)], iv: (u32, u64, u64)) -> Segment {
    let (lo, hi) = gene_range(genome, runs, iv.0, iv.1, iv.2);
    Segment {
        bulk: None,
        chrom: iv.0,
        seg: 0,
        lo,
        hi,
        start: iv.1,
        end: iv.2,
        n_genes: hi - lo,
        state: CnState::Neu,
        retest: None,
    }
}

/// Consensus segments across bulks (SPEC 6.5, `get_segs_consensus` with
/// `retest = FALSE` and `fill_neu_segs`, N/R/main.R:870-1013).
///
/// 1. Segments with `start == end` are dropped; untested ones or those with
///    `LLR < min_llr` count as `neu`.
/// 2. [`resolve_cnvs`] on the rest.
/// 3. The bp union of `neu` segments, minus the kept CNVs, fills the gaps
///    (pieces with `end - start > 0` only).
///
/// Every output segment's gene range is the genes inside its bp interval.
/// Kept CNVs keep their bulk, label, retest and `n_genes`.
///
/// ### Params
///
/// * `segs` - Segments of all bulks, retested (SPEC 6.4)
/// * `genome` - Gene annotation the gene indices refer to
/// * `params` - Consensus parameters
///
/// ### Returns
///
/// Consensus segments ordered by chromosome then start, numbered per
/// chromosome, or an input error.
pub fn segs_consensus(
    segs: &[Segment],
    genome: &Genome,
    params: &ConsensusParams,
) -> Result<Vec<Segment>, CnvErrors> {
    let runs = genome_runs(genome)?;
    let mut cnv = Vec::new();
    let mut neu: Ranges = Vec::new();
    for s in segs.iter().filter(|s| s.start != s.end) {
        let llr = llrs(s).0;
        if s.state == CnState::Neu || llr.is_nan() || llr < params.min_llr {
            neu.push((s.chrom, s.start, s.end));
        } else {
            cnv.push(*s);
        }
    }
    let neu = reduce_ranges(neu);
    let mut out: Vec<Segment> = if cnv.is_empty() {
        neu.into_iter()
            .map(|iv| neu_segment(genome, &runs, iv))
            .collect()
    } else {
        let star = resolve_cnvs(&cnv, params.min_overlap);
        let star_iv: Ranges = star.iter().map(|s| (s.chrom, s.start, s.end)).collect();
        let gaps = setdiff_ranges(neu, star_iv)
            .into_iter()
            .filter(|iv| iv.2 > iv.1);
        let mut all: Vec<Segment> = star
            .into_iter()
            .map(|mut s| {
                (s.lo, s.hi) = gene_range(genome, &runs, s.chrom, s.start, s.end);
                s
            })
            .collect();
        all.extend(gaps.map(|iv| neu_segment(genome, &runs, iv)));
        all.sort_by_key(|s| (s.chrom, s.start));
        all
    };
    let mut prev = None;
    let mut k = 0;
    for s in &mut out {
        if prev != Some(s.chrom) {
            k = 0;
            prev = Some(s.chrom);
        }
        k += 1;
        s.seg = k;
    }
    Ok(out)
}

/// Whether the baseline includes LOH (SPEC 6.6): the `neu` consensus spans
/// fewer than 1.5e8 bp, summing `end - start`.
///
/// ### Params
///
/// * `consensus` - Consensus segments
///
/// ### Returns
///
/// `use_loh`.
pub fn use_loh(consensus: &[Segment]) -> bool {
    let bp: f64 = consensus
        .iter()
        .filter(|s| s.state == CnState::Neu)
        .map(|s| (s.end - s.start) as f64)
        .sum();
    bp < USE_LOH_MAX_NEU_BP
}

/// Retest consensus segments on one bulk (SPEC 6.6, `retest_bulks`).
///
/// Genes map to the first consensus segment covering them; unmapped genes
/// are dropped. The baseline is re-fitted on genes in `neu` segments (plus
/// `loh_*` when `use_loh`), then every non-`neu` segment is retested and its
/// state set to `neu` if `LLR < min_llr`. `neu` segments stay untested,
/// which is numbat's `LLR = 0`, `cnv_state_post = neu`.
///
/// ### Params
///
/// * `bulk` - The bulk
/// * `bulk_id` - Id stamped on the output
/// * `consensus` - Consensus segments (SPEC 6.5)
/// * `use_loh` - See [`use_loh`]
/// * `logphi_min` - `log2` fold change of a single-copy event
/// * `min_llr` - `LLR` threshold
///
/// ### Returns
///
/// One segment per consensus segment, in the same order, or an error.
pub fn retest_consensus<T: CnvFloat>(
    bulk: &BulkGenes<T>,
    bulk_id: u32,
    consensus: &[Segment],
    use_loh: bool,
    logphi_min: f64,
    min_llr: f64,
) -> Result<Vec<Segment>, CnvErrors> {
    check_bulk(bulk)?;
    let owner = assign_genes(bulk.gene, consensus);
    let mapped: Vec<usize> = (0..owner.len()).filter(|&g| owner[g].is_some()).collect();
    let y: Vec<u32> = mapped.iter().map(|&g| bulk.y[g]).collect();
    let l: Vec<T> = mapped.iter().map(|&g| bulk.lambda[g]).collect();
    let diploid: Vec<bool> = mapped
        .iter()
        .map(|&g| {
            let st = consensus[owner[g].expect("mapped")].state;
            st == CnState::Neu || (use_loh && st.is_loh())
        })
        .collect();
    if y.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "bulk genes inside consensus segments",
        });
    }
    let base = fit_baseline(&y, &l, bulk.d, &diploid)?;
    let mut out: Vec<Segment> = consensus
        .iter()
        .map(|s| Segment {
            bulk: Some(bulk_id),
            retest: None,
            ..*s
        })
        .collect();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); consensus.len()];
    for &g in &mapped {
        members[owner[g].expect("mapped")].push(g);
    }
    out.par_iter_mut().zip(&members).try_for_each(|(seg, m)| {
        if seg.state == CnState::Neu || m.is_empty() {
            return Ok(());
        }
        let ys: Vec<u32> = m.iter().map(|&g| bulk.y[g]).collect();
        let ls: Vec<T> = m.iter().map(|&g| bulk.lambda[g]).collect();
        let mut r = retest_segment(&ys, &ls, bulk.d, base.mu, base.sigma, logphi_min)?;
        if r.llr < min_llr {
            r.state_post = PostState::Neu;
        }
        seg.retest = Some(r);
        Ok(())
    })?;
    Ok(out)
}

/// Multi-state segments (SPEC 6.7, `test_multi_allelic`,
/// N/R/main.R:1682-1733).
///
/// Per consensus segment, the distinct `state_post` among leaf bulks with
/// `LLR > min_llr` and max non-neutral probability `> p_min`. With more than
/// one, the segment's state probabilities become 0.5 for each present state
/// and 0 otherwise.
///
/// ### Params
///
/// * `consensus` - Consensus segments, updated in place
/// * `leaves` - Per leaf bulk, [`retest_consensus`] output aligned to
///   `consensus`
/// * `min_llr` - `LLR` threshold
/// * `p_min` - Probability threshold, numbat `1 - 1e-4`
///
/// ### Returns
///
/// Per consensus segment, the sorted states if it is multi-state.
pub fn test_multi_allelic(
    consensus: &mut [Segment],
    leaves: &[Vec<Segment>],
    min_llr: f64,
    p_min: f64,
) -> Result<Vec<Option<Vec<PostState>>>, CnvErrors> {
    for l in leaves {
        check_lens(consensus.len(), &[("leaf segments", l.len())])?;
    }
    let mut out = Vec::with_capacity(consensus.len());
    for (s, seg) in consensus.iter_mut().enumerate() {
        let mut states: Vec<PostState> = leaves
            .iter()
            .filter_map(|l| l[s].retest)
            .filter(|r| {
                let p_max = r
                    .p_amp
                    .max(r.p_del)
                    .max(r.p_bdel)
                    .max(r.p_loh)
                    .max(r.p_bamp);
                r.llr > min_llr && p_max > p_min
            })
            .map(|r| r.state_post)
            .collect();
        states.sort_unstable();
        states.dedup();
        if states.len() > 1 {
            if let Some(r) = seg.retest.as_mut() {
                let pr = |p: PostState| {
                    if states.contains(&p) {
                        MULTI_STATE_PRIOR
                    } else {
                        0.0
                    }
                };
                r.p_amp = pr(PostState::Amp);
                r.p_del = pr(PostState::Del);
                r.p_loh = pr(PostState::Loh);
                r.p_bamp = pr(PostState::Bamp);
                r.p_bdel = pr(PostState::Bdel);
            }
            out.push(Some(states));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixtures");

    /// Rows of a TSV fixture, header dropped.
    fn read_tsv(name: &str) -> Vec<Vec<String>> {
        let s = std::fs::read_to_string(format!("{FIXTURES}/{name}"))
            .unwrap_or_else(|e| panic!("{name}: {e}; run reference/segments.R"));
        s.lines()
            .skip(1)
            .map(|l| l.split('\t').map(|c| c.trim().to_string()).collect())
            .collect()
    }

    /// Parse an R number, `NA` as `NaN`.
    fn num(s: &str) -> f64 {
        match s {
            "NA" => f64::NAN,
            "-Inf" => f64::NEG_INFINITY,
            "Inf" => f64::INFINITY,
            _ => s.parse().unwrap(),
        }
    }

    /// numbat's copy number label.
    fn cn(s: &str) -> CnState {
        match s {
            "neu" => CnState::Neu,
            "del_1" => CnState::Del1,
            "del_2" => CnState::Del2,
            "loh_1" => CnState::Loh1,
            "loh_2" => CnState::Loh2,
            "amp_1" => CnState::Amp1,
            "amp_2" => CnState::Amp2,
            "bamp" => CnState::Bamp,
            "bdel" => CnState::Bdel,
            _ => panic!("{s}"),
        }
    }

    /// numbat's retest label.
    fn post(s: &str) -> PostState {
        match s {
            "neu" => PostState::Neu,
            "loh" => PostState::Loh,
            "amp" => PostState::Amp,
            "del" => PostState::Del,
            "bamp" => PostState::Bamp,
            "bdel" => PostState::Bdel,
            _ => panic!("{s}"),
        }
    }

    /// Inverse of numbat's `generate_postfix`.
    fn ordinal(s: &str) -> u32 {
        s.bytes().fold(0, |a, b| a * 26 + u32::from(b - b'a' + 1))
    }

    /// Absolute-or-relative closeness.
    fn close(a: f64, b: f64, tol: f64) -> bool {
        a == b || (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn test_annot_and_smooth_segs_vs_numbat() {
        let rows = read_tsv("segs_annot.tsv");
        let chrom: Vec<u32> = rows.iter().map(|r| r[0].parse().unwrap()).collect();
        let pos: Vec<u64> = rows.iter().map(|r| r[1].parse().unwrap()).collect();
        let gene: Vec<u32> = rows.iter().map(|r| r[2].parse().unwrap()).collect();
        let states: Vec<CnState> = rows.iter().map(|r| cn(&r[3])).collect();

        let check = |segs: &[Segment], col: usize| {
            for (g, r) in rows.iter().enumerate() {
                let s = segs
                    .iter()
                    .find(|s| s.chrom == chrom[g] && gene[g] >= s.lo && gene[g] < s.hi)
                    .unwrap();
                assert_eq!(s.seg, ordinal(&r[col]), "gene {g}");
                assert_eq!(s.start, r[col + 1].parse::<u64>().unwrap(), "gene {g}");
                assert_eq!(s.end, r[col + 2].parse::<u64>().unwrap(), "gene {g}");
                assert_eq!(s.n_genes, r[col + 3].parse::<u32>().unwrap(), "gene {g}");
            }
        };
        let a1 = annot_segs(Some(0), &chrom, &pos, &gene, &states).unwrap();
        check(&a1, 4);
        assert!(a1.iter().all(|s| s.bulk == Some(0) && s.retest.is_none()));

        let sm = smooth_segs(&chrom, &states, 10).unwrap();
        let want: Vec<CnState> = rows.iter().map(|r| cn(&r[8])).collect();
        assert_eq!(sm, want);
        let a2 = annot_segs(Some(0), &chrom, &pos, &gene, &sm).unwrap();
        check(&a2, 9);
        for s in &a2 {
            let g = gene.iter().position(|&x| x == s.lo).unwrap();
            assert_eq!(s.state, sm[g]);
        }
    }

    #[test]
    fn test_smooth_segs_all_short_chromosome_d3() {
        // numbat errors here; D3 sets the chromosome to neu
        let chrom = [1, 1, 1, 1, 2, 2, 2, 2, 2];
        let states = [
            CnState::Amp1,
            CnState::Amp1,
            CnState::Del1,
            CnState::Del1,
            CnState::Amp1,
            CnState::Amp1,
            CnState::Amp1,
            CnState::Del1,
            CnState::Del1,
        ];
        let sm = smooth_segs(&chrom, &states, 2).unwrap();
        assert_eq!(&sm[..4], &[CnState::Neu; 4]);
        assert_eq!(&sm[4..], &[CnState::Amp1; 5]);
    }

    #[test]
    fn test_retest_maths_vs_numbat() {
        for r in read_tsv("segs_retest_grid.tsv") {
            let (phi, s) = (num(&r[0]), num(&r[1]));
            let ours = retest_raw(phi, s, 0.25);
            let want = [
                num(&r[3]),
                num(&r[4]),
                num(&r[5]),
                num(&r[6]),
                num(&r[7]),
                num(&r[8]),
            ];
            let got = [
                ours.p_neu,
                ours.p_loh,
                ours.p_amp,
                ours.p_del,
                ours.p_bamp,
                ours.p_bdel,
            ];
            assert!(
                close(ours.llr, num(&r[2]), 1e-10),
                "phi {phi} s {s}: llr {} vs {}",
                ours.llr,
                r[2]
            );
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (g - w).abs() <= 1e-10,
                    "phi {phi} s {s}: {got:?} vs {want:?}"
                );
            }
            assert_eq!(ours.state_post, post(&r[9]), "phi {phi} s {s}");
        }
    }

    #[test]
    fn test_retest_d5_folds_states_and_probs() {
        for r in read_tsv("segs_retest_grid.tsv") {
            let raw = retest_raw(num(&r[0]), num(&r[1]), 0.25);
            let f = fold_d5(raw);
            let want = match raw.state_post {
                PostState::Loh => PostState::Neu,
                PostState::Bamp => PostState::Amp,
                PostState::Bdel => PostState::Del,
                s => s,
            };
            assert_eq!(f.state_post, want);
            assert!(matches!(
                f.state_post,
                PostState::Neu | PostState::Amp | PostState::Del
            ));
            assert_eq!(f.p_amp, raw.p_amp + raw.p_bamp);
            assert_eq!(f.p_del, raw.p_del + raw.p_bdel);
            assert_eq!(
                (f.p_bamp, f.p_bdel, f.p_loh, f.llr),
                (0.0, 0.0, raw.p_loh, raw.llr)
            );
        }
    }

    #[test]
    fn test_retest_segment_vs_numbat() {
        let rows = read_tsv("segs_retest_genes.tsv");
        let one = &read_tsv("segs_retest_one.tsv")[0];
        let (d, mu, sigma, llr_x) = (num(&one[0]), num(&one[1]), num(&one[2]), num(&one[3]));
        let (mut y, mut l) = (Vec::new(), Vec::new());
        for r in rows.iter().filter(|r| r[2] == "TRUE") {
            y.push(r[0].parse::<u32>().unwrap());
            l.push(num(&r[1]));
        }
        let ours = retest_segment(&y, &l, d, mu, sigma, 0.25).unwrap();
        assert!(
            close(ours.phi_hat, num(&one[4]), 1e-5),
            "{} vs {}",
            ours.phi_hat,
            one[4]
        );
        assert!(
            close(ours.s_phi, num(&one[5]), 1e-4),
            "{} vs {}",
            ours.s_phi,
            one[5]
        );
        assert!(
            close(ours.llr, num(&one[6]), 1e-4),
            "{} vs {}",
            ours.llr,
            one[6]
        );
        assert!(close(ours.llr_x, llr_x, 1e-6), "{} vs {llr_x}", ours.llr_x);
        let p_amp = num(&one[9]) + num(&one[11]);
        assert!(
            (ours.p_amp - p_amp).abs() < 1e-6,
            "{} vs {p_amp}",
            ours.p_amp
        );
        assert!((ours.p_neu - num(&one[7])).abs() < 1e-6);
        // numbat: amp (first of the amp/bamp tie); D5 keeps it amp
        assert_eq!(post(&one[13]), PostState::Amp);
        assert_eq!(ours.state_post, PostState::Amp);
    }

    /// Fixture segments and genome for the consensus tests.
    fn consensus_inputs(all_neu: bool) -> (Vec<Segment>, Vec<u32>, Vec<u64>) {
        let g = read_tsv("segs_cons_genes.tsv");
        let chrom = g.iter().map(|r| r[0].parse().unwrap()).collect();
        let pos = g.iter().map(|r| r[1].parse().unwrap()).collect();
        let segs = read_tsv("segs_cons_in.tsv")
            .iter()
            .map(|r| {
                let llr = num(&r[8]);
                let retest = (!llr.is_nan()).then(|| Retest {
                    phi_hat: 1.0,
                    s_phi: 0.0,
                    llr: if all_neu { 1.0 } else { llr },
                    llr_x: num(&r[9]),
                    p_neu: 0.0,
                    p_loh: 0.0,
                    p_amp: 0.0,
                    p_del: 0.0,
                    p_bamp: 0.0,
                    p_bdel: 0.0,
                    state_post: PostState::Amp,
                });
                Segment {
                    bulk: Some(r[0].parse().unwrap()),
                    chrom: r[1].parse().unwrap(),
                    seg: 0,
                    lo: r[2].parse().unwrap(),
                    hi: r[3].parse().unwrap(),
                    start: r[4].parse().unwrap(),
                    end: r[5].parse().unwrap(),
                    n_genes: r[6].parse().unwrap(),
                    state: cn(&r[7]),
                    retest,
                }
            })
            .collect();
        (segs, chrom, pos)
    }

    #[test]
    fn test_segs_consensus_vs_numbat() {
        let (segs, chrom, pos) = consensus_inputs(false);
        let genome = Genome {
            chrom: &chrom,
            pos: &pos,
        };
        let out = segs_consensus(&segs, &genome, &ConsensusParams::default()).unwrap();
        let want = read_tsv("segs_cons_out.tsv");
        assert_eq!(out.len(), want.len());
        for (s, r) in out.iter().zip(&want) {
            let bulk = if r[0] == "NA" {
                None
            } else {
                Some(r[0].parse().unwrap())
            };
            assert_eq!(s.bulk, bulk, "{r:?}");
            assert_eq!(s.chrom, r[1].parse::<u32>().unwrap());
            assert_eq!(
                (s.start, s.end),
                (r[2].parse().unwrap(), r[3].parse().unwrap()),
                "{r:?}"
            );
            assert_eq!(s.state, cn(&r[4]), "{r:?}");
            assert_eq!(s.seg, ordinal(&r[5]));
            // gene range is the genes inside the bp interval
            for g in s.lo as usize..s.hi as usize {
                assert!(chrom[g] == s.chrom && pos[g] >= s.start && pos[g] <= s.end);
            }
        }
    }

    #[test]
    fn test_segs_consensus_all_neu_vs_numbat() {
        let (segs, chrom, pos) = consensus_inputs(true);
        let genome = Genome {
            chrom: &chrom,
            pos: &pos,
        };
        let out = segs_consensus(&segs, &genome, &ConsensusParams::default()).unwrap();
        let want = read_tsv("segs_cons_allneu.tsv");
        assert_eq!(out.len(), want.len());
        for (s, r) in out.iter().zip(&want) {
            assert_eq!(s.chrom, r[0].parse::<u32>().unwrap());
            assert_eq!(
                (s.start, s.end),
                (r[1].parse().unwrap(), r[2].parse().unwrap())
            );
            assert_eq!(
                (s.state, s.bulk, s.seg),
                (CnState::Neu, None, ordinal(&r[3]))
            );
        }
    }

    #[test]
    fn test_multi_allelic_vs_numbat() {
        let cons_rows = read_tsv("segs_multi_cons.tsv");
        let blank = |p: [f64; 5], st: PostState, llr: f64| Retest {
            phi_hat: 1.0,
            s_phi: 0.0,
            llr,
            llr_x: 0.0,
            p_neu: 0.0,
            p_amp: p[0],
            p_del: p[1],
            p_bdel: p[2],
            p_loh: p[3],
            p_bamp: p[4],
            state_post: st,
        };
        let seg = |chrom: u32, state: CnState, retest: Option<Retest>| Segment {
            bulk: None,
            chrom,
            seg: 0,
            lo: 0,
            hi: 0,
            start: 0,
            end: 0,
            n_genes: 0,
            state,
            retest,
        };
        let names: Vec<String> = cons_rows.iter().map(|r| r[1].clone()).collect();
        let mut cons: Vec<Segment> = cons_rows
            .iter()
            .map(|r| {
                let st = cn(&r[2]);
                let p = [num(&r[4]), num(&r[5]), num(&r[6]), num(&r[7]), num(&r[8])];
                let rt = (st != CnState::Neu).then(|| blank(p, post(&r[3]), 50.0));
                seg(r[0].parse().unwrap(), st, rt)
            })
            .collect();
        let bulk_rows = read_tsv("segs_multi_bulks.tsv");
        let mut leaves: Vec<Vec<Segment>> = Vec::new();
        for b in 1..=3usize {
            leaves.push(
                cons.iter()
                    .map(|c| Segment { retest: None, ..*c })
                    .collect(),
            );
            for r in bulk_rows.iter().filter(|r| r[0] == b.to_string()) {
                let i = names.iter().position(|n| *n == r[2]).unwrap();
                let p = [num(&r[4]), num(&r[5]), num(&r[6]), num(&r[7]), num(&r[8])];
                leaves[b - 1][i].retest = Some(blank(p, post(&r[9]), num(&r[3])));
            }
        }
        let multi = test_multi_allelic(&mut cons, &leaves, 5.0, 1.0 - 1e-4).unwrap();
        for (i, r) in read_tsv("segs_multi_out.tsv").iter().enumerate() {
            let n: usize = r[1].parse().unwrap();
            if n > 1 {
                let want: Vec<PostState> = r[2].split(',').map(post).collect();
                assert_eq!(multi[i].as_deref(), Some(want.as_slice()), "{r:?}");
            } else {
                assert!(multi[i].is_none(), "{r:?}");
            }
            if let Some(rt) = cons[i].retest {
                let got = [rt.p_amp, rt.p_del, rt.p_bdel, rt.p_loh, rt.p_bamp];
                for (k, g) in got.iter().enumerate() {
                    assert_eq!(*g, num(&r[3 + k]), "{r:?}");
                }
            } else {
                assert_eq!(r[3], "NA");
            }
        }
    }

    #[test]
    fn test_retest_consensus_finds_planted_gain() {
        // ours (D4): numbat has no expression-only retest to compare with
        let rows = read_tsv("segs_retest_genes.tsv");
        let d = num(&read_tsv("segs_retest_one.tsv")[0][0]);
        let y: Vec<u32> = rows.iter().map(|r| r[0].parse().unwrap()).collect();
        let l: Vec<f64> = rows.iter().map(|r| num(&r[1])).collect();
        let n = y.len() as u32;
        let gene: Vec<u32> = (0..n).collect();
        let chrom = vec![1u32; y.len()];
        let pos: Vec<u64> = (0..u64::from(n)).map(|i| 1000 * (i + 1)).collect();
        let states: Vec<CnState> = (0..n)
            .map(|i| if i < 250 { CnState::Neu } else { CnState::Amp1 })
            .collect();
        let cons = annot_segs(None, &chrom, &pos, &gene, &states).unwrap();
        let bulk = BulkGenes {
            gene: &gene,
            y: &y,
            lambda: &l,
            d,
        };
        assert!(use_loh(&cons));
        let out = retest_consensus(&bulk, 7, &cons, false, 0.25, 5.0).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].retest.is_none() && out[0].bulk == Some(7));
        let r = out[1].retest.unwrap();
        assert_eq!(r.state_post, PostState::Amp);
        assert!(r.llr > 5.0 && r.phi_hat > 1.2, "{r:?}");

        // first pass: the same segment through retest_segments
        let base = fit_baseline(&y, &l, d, &vec![true; y.len()]).unwrap();
        let mut segs = cons.clone();
        retest_segments(&bulk, &mut segs, &base, 0.25).unwrap();
        assert!(segs[0].retest.is_none());
        assert_eq!(segs[1].retest.unwrap().state_post, PostState::Amp);
    }

    #[test]
    fn test_ln_pnorm_tails_continuous() {
        // series just below the cut-over against erfc at the same point
        let x = LN_PNORM_ASYMPTOTIC_BELOW - 1e-9;
        let a = ln_pnorm(x);
        let b = (0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)).ln();
        assert!(close(a, b, 1e-13), "{a} vs {b}");
        assert_eq!(ln_pnorm(0.0), 0.5f64.ln());
        assert_eq!(pnorm_range_log(0.1, 3.0, 1.0, 0.0), 1.0);
    }
}
