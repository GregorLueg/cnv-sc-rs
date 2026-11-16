//! HMM engine and the 15-state expression model (SPEC 5).
//!
//! Expression only (`p_s = 0`), so the transition matrix is constant along a
//! sequence. Emissions are the PLN density at five distinct `phi` values,
//! computed once per gene and scattered to the 15 states. One Viterbi is
//! sequential; independent sequences (chromosomes, bulks) run in parallel.

use rayon::prelude::*;

use crate::errors::CnvErrors;
use crate::model::pln::ln_pln;
use crate::utils::traits::{CnvFloat, wide};

////////////
// States //
////////////

/// Number of HMM states (SPEC 5.1).
pub const N_STATES: usize = 15;

/// Number of distinct `phi` values across the states (SPEC 5.1).
pub const N_PHI: usize = 5;

/// Copy number label of a state, `cnv_state` in numbat: the state name with
/// `_up`/`_down` stripped (H/R/hahmmr.R:568).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CnState {
    /// Neutral
    Neu,
    /// Single-copy deletion
    Del1,
    /// Two-copy deletion
    Del2,
    /// Copy-neutral LOH
    Loh1,
    /// Two-copy LOH
    Loh2,
    /// Single-copy gain
    Amp1,
    /// Two-copy gain
    Amp2,
    /// Balanced gain
    Bamp,
    /// Balanced deletion
    Bdel,
}

impl CnState {
    /// Relative abundance `w` of the copy number state (H/R/hahmmr.R:573).
    ///
    /// ### Returns
    ///
    /// The weight used in the off-diagonal transitions.
    pub const fn weight(self) -> f64 {
        match self {
            CnState::Neu | CnState::Del1 | CnState::Loh1 | CnState::Amp1 => 1.0,
            CnState::Bamp => 1e-4,
            CnState::Del2 | CnState::Loh2 | CnState::Amp2 | CnState::Bdel => 1e-10,
        }
    }

    /// Whether the label is copy-neutral LOH (`loh_1` or `loh_2`).
    ///
    /// ### Returns
    ///
    /// `true` for the LOH labels.
    pub const fn is_loh(self) -> bool {
        matches!(self, CnState::Loh1 | CnState::Loh2)
    }
}

/// All nine copy number labels, in hahmmr's order of `w`.
const CN_ALL: [CnState; 9] = [
    CnState::Neu,
    CnState::Del1,
    CnState::Del2,
    CnState::Loh1,
    CnState::Loh2,
    CnState::Amp1,
    CnState::Amp2,
    CnState::Bamp,
    CnState::Bdel,
];

/// Haplotype phase of a state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// No phase (`neu`, `bamp`, `bdel`)
    Unphased,
    /// `_up`
    Up,
    /// `_down`
    Down,
}

/// One HMM state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HmmState {
    /// Copy number label
    pub cn: CnState,
    /// Phase
    pub phase: Phase,
    /// Column of the distinct-`phi` emission table, see [`state_phis`]
    pub phi_col: usize,
}

/// Shorthand for the state table.
const fn st(cn: CnState, phase: Phase, phi_col: usize) -> HmmState {
    HmmState { cn, phase, phi_col }
}

/// The 15 states in hahmmr's order (H/R/hahmmr.R:565-570, 626).
pub const STATES: [HmmState; N_STATES] = [
    st(CnState::Neu, Phase::Unphased, 0),
    st(CnState::Del1, Phase::Up, 1),
    st(CnState::Del1, Phase::Down, 1),
    st(CnState::Del2, Phase::Up, 2),
    st(CnState::Del2, Phase::Down, 2),
    st(CnState::Loh1, Phase::Up, 0),
    st(CnState::Loh1, Phase::Down, 0),
    st(CnState::Loh2, Phase::Up, 0),
    st(CnState::Loh2, Phase::Down, 0),
    st(CnState::Amp1, Phase::Up, 3),
    st(CnState::Amp1, Phase::Down, 3),
    st(CnState::Amp2, Phase::Up, 4),
    st(CnState::Amp2, Phase::Down, 4),
    st(CnState::Bamp, Phase::Unphased, 3),
    st(CnState::Bdel, Phase::Unphased, 1),
];

/// `phi` of a two-copy deletion (H/R/hahmmr.R:626).
const PHI_DEL2: f64 = 0.5;
/// `phi` of a two-copy gain (H/R/hahmmr.R:626).
const PHI_AMP2: f64 = 2.5;

/// The five distinct `phi` values, indexed by [`HmmState::phi_col`].
///
/// ### Params
///
/// * `logphi_min` - `log2` fold change of a single-copy event (numbat 0.25)
///
/// ### Returns
///
/// `[1, 2^-logphi_min, 0.5, 2^logphi_min, 2.5]`.
pub fn state_phis(logphi_min: f64) -> [f64; N_PHI] {
    [
        1.0,
        (-logphi_min).exp2(),
        PHI_DEL2,
        logphi_min.exp2(),
        PHI_AMP2,
    ]
}

////////////
// Params //
////////////

/// Parameters of the expression HMM.
#[derive(Clone, Copy, Debug)]
pub struct HmmParams {
    /// Transition probability between copy number states (SPEC 5.2)
    pub t: f64,
    /// `log2` fold change of a single-copy event (SPEC 5.1)
    pub logphi_min: f64,
}

impl HmmParams {
    /// Create parameters.
    ///
    /// ### Params
    ///
    /// * `t` - Transition probability, in `(0, 1]`
    /// * `logphi_min` - `log2` fold change of a single-copy event, `> 0`
    ///
    /// ### Returns
    ///
    /// The parameter struct.
    pub fn new(t: f64, logphi_min: f64) -> Self {
        Self { t, logphi_min }
    }
}

impl Default for HmmParams {
    /// numbat defaults, N/R/main.R:68-77.
    fn default() -> Self {
        Self {
            t: 1e-5,
            logphi_min: 0.25,
        }
    }
}

/////////////////
// Transitions //
/////////////////

/// Transition probability between two states, `p_s = 0`
/// (`get_trans_probs_s15`, H/R/hahmmr.R:700-728).
///
/// ### Params
///
/// * `t` - Transition probability between copy number states
/// * `from` - Origin state
/// * `to` - Destination state
///
/// ### Returns
///
/// The probability (not log).
fn trans_prob(t: f64, from: HmmState, to: HmmState) -> f64 {
    if from.cn == to.cn {
        if from.phase == to.phase { 1.0 - t } else { 0.0 }
    } else {
        // sum of w over the other copy number states, in hahmmr's order
        let denom: f64 = CN_ALL
            .iter()
            .filter(|&&c| c != from.cn)
            .map(|c| c.weight())
            .sum();
        let p = t * to.cn.weight() / denom;
        if to.phase == Phase::Unphased {
            p
        } else {
            p / 2.0
        }
    }
}

/// Check `t` lies in `(0, 1]`.
///
/// ### Params
///
/// * `t` - Transition probability
///
/// ### Returns
///
/// `Ok(())` or an `InvalidParameter` error.
fn check_t(t: f64) -> Result<(), CnvErrors> {
    if t > 0.0 && t <= 1.0 {
        Ok(())
    } else {
        Err(CnvErrors::InvalidParameter {
            name: "t",
            value: t.to_string(),
            expected: "in (0, 1]",
        })
    }
}

/// Log transition matrix (SPEC 5.2, `calc_trans_mat_s15` with `p_s = 0`).
///
/// ### Params
///
/// * `t` - Transition probability between copy number states, in `(0, 1]`
///
/// ### Returns
///
/// `N_STATES x N_STATES` row-major, `[from * N_STATES + to]`; impossible
/// transitions are `-inf`.
pub fn log_transitions(t: f64) -> Result<Vec<f64>, CnvErrors> {
    check_t(t)?;
    let mut a = Vec::with_capacity(N_STATES * N_STATES);
    for from in STATES {
        for to in STATES {
            a.push(trans_prob(t, from, to).ln());
        }
    }
    Ok(a)
}

/// Log initial distribution (SPEC 5.3): transitions out of `neu` with
/// `t' = min(100 t, 1)`.
///
/// ### Params
///
/// * `t` - Transition probability between copy number states, in `(0, 1]`
///
/// ### Returns
///
/// `ln delta` per state.
pub fn log_initial(t: f64) -> Result<[f64; N_STATES], CnvErrors> {
    check_t(t)?;
    let t0 = (100.0 * t).min(1.0);
    let mut d = [0.0; N_STATES];
    for (j, to) in STATES.iter().enumerate() {
        d[j] = trans_prob(t0, STATES[0], *to).ln();
    }
    Ok(d)
}

/////////////
// Viterbi //
/////////////

/// Log-space Viterbi (SPEC 5.5, H/src/hmm.cpp:151-197).
///
/// The forward pass stores the argmax predecessor per step (one byte per
/// state). Ties go to the lowest index, which is the same predecessor
/// hahmmr's backtrack recomputes.
///
/// ### Params
///
/// * `log_delta` - Log initial distribution, length `M`
/// * `log_a` - Log transitions, `M x M` row-major `[from * M + to]`
/// * `log_e` - Log emissions, `n x M` row-major
/// * `m` - Number of states, `1..=256`
///
/// ### Returns
///
/// The most probable state per position, or a shape error.
pub fn viterbi(
    log_delta: &[f64],
    log_a: &[f64],
    log_e: &[f64],
    m: usize,
) -> Result<Vec<u8>, CnvErrors> {
    if m == 0 || m > 256 {
        return Err(CnvErrors::InvalidParameter {
            name: "m",
            value: m.to_string(),
            expected: "in 1..=256",
        });
    }
    if log_delta.len() != m {
        return Err(CnvErrors::ShapeMismatch {
            what: "log_delta",
            expected: m,
            got: log_delta.len(),
        });
    }
    if log_a.len() != m * m {
        return Err(CnvErrors::ShapeMismatch {
            what: "log_a",
            expected: m * m,
            got: log_a.len(),
        });
    }
    if !log_e.len().is_multiple_of(m) {
        return Err(CnvErrors::ShapeMismatch {
            what: "log_e (multiple of m)",
            expected: m * (log_e.len() / m + 1),
            got: log_e.len(),
        });
    }
    let n = log_e.len() / m;
    if n == 0 {
        return Ok(Vec::new());
    }

    // to-major copy so the inner max runs over a contiguous column
    let mut at = vec![0.0; m * m];
    for k in 0..m {
        for j in 0..m {
            at[j * m + k] = log_a[k * m + j];
        }
    }

    let mut nu: Vec<f64> = log_delta
        .iter()
        .zip(&log_e[..m])
        .map(|(d, e)| d + e)
        .collect();
    let mut next = vec![0.0; m];
    let mut back = vec![0u8; n * m];
    for i in 1..n {
        let e = &log_e[i * m..(i + 1) * m];
        let bp = &mut back[i * m..(i + 1) * m];
        for j in 0..m {
            let col = &at[j * m..(j + 1) * m];
            let mut best = nu[0] + col[0];
            let mut arg = 0;
            for k in 1..m {
                let v = nu[k] + col[k];
                if v > best {
                    best = v;
                    arg = k;
                }
            }
            next[j] = best + e[j];
            bp[j] = arg as u8;
        }
        std::mem::swap(&mut nu, &mut next);
    }

    let mut z = vec![0u8; n];
    let mut arg = 0;
    for k in 1..m {
        if nu[k] > nu[arg] {
            arg = k;
        }
    }
    z[n - 1] = arg as u8;
    for i in (0..n - 1).rev() {
        z[i] = back[(i + 1) * m + z[i + 1] as usize];
    }
    Ok(z)
}

///////////////
// Emissions //
///////////////

/// One pseudobulk as the HMM sees it (SPEC 5.4). Genes in genomic order.
#[derive(Clone, Copy, Debug)]
pub struct BulkHmmInput<'a, T: CnvFloat> {
    /// Chromosome per gene, contiguous runs
    pub chrom: &'a [u32],
    /// Pseudobulk counts per gene
    pub y: &'a [u32],
    /// Reference proportions per gene (the bulk's `lambda_bar`), all `> 0`
    pub lambda: &'a [T],
    /// Library size over all annotated genes
    pub d: f64,
    /// Baseline `mu` (SPEC 6.1)
    pub mu: f64,
    /// Baseline `sigma` (SPEC 6.1), `> 0`
    pub sigma: f64,
}

/// Log emissions for a run of genes (SPEC 5.4).
///
/// Five PLN evaluations per gene, one per distinct `phi`, scattered to the
/// 15 states.
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, same length, all `> 0`
/// * `d` - Library size
/// * `mu` - Baseline `mu`
/// * `sigma` - Baseline `sigma`
/// * `logphi_min` - `log2` fold change of a single-copy event
///
/// ### Returns
///
/// `n x N_STATES` row-major log emissions.
pub fn emission_log_probs<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    mu: f64,
    sigma: f64,
    logphi_min: f64,
) -> Vec<f64> {
    let ln_phi = state_phis(logphi_min).map(f64::ln);
    let mut out = vec![0.0; y.len() * N_STATES];
    for ((row, &yg), &lg) in out
        .as_chunks_mut::<N_STATES>()
        .0
        .iter_mut()
        .zip(y)
        .zip(lambda)
    {
        let base = mu + (d * wide(lg)).ln();
        let mut col = [0.0; N_PHI];
        for (c, lp) in col.iter_mut().zip(&ln_phi) {
            *c = ln_pln(yg, base + lp, sigma);
        }
        for (r, s) in row.iter_mut().zip(&STATES) {
            *r = col[s.phi_col];
        }
    }
    out
}

/// Contiguous gene ranges per chromosome.
///
/// ### Params
///
/// * `chrom` - Chromosome per gene
///
/// ### Returns
///
/// `[lo, hi)` per run of equal ids, or an error if a chromosome reappears
/// after another one (not genomic order).
pub(crate) fn chrom_runs(chrom: &[u32]) -> Result<Vec<(usize, usize)>, CnvErrors> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0;
    for i in 1..=chrom.len() {
        if i == chrom.len() || chrom[i] != chrom[lo] {
            if runs.iter().any(|&(a, _)| chrom[a] == chrom[lo]) {
                return Err(CnvErrors::NotGenomicOrder {
                    what: "chromosome ids are not contiguous",
                });
            }
            runs.push((lo, i));
            lo = i;
        }
    }
    Ok(runs)
}

/// Decode one pseudobulk: emissions and Viterbi per chromosome, chromosomes
/// in parallel (SPEC 5).
///
/// ### Params
///
/// * `bulk` - The pseudobulk
/// * `params` - HMM parameters
///
/// ### Returns
///
/// State index into [`STATES`] per gene, or an input error.
pub fn decode_bulk<T: CnvFloat>(
    bulk: &BulkHmmInput<T>,
    params: &HmmParams,
) -> Result<Vec<u8>, CnvErrors> {
    let n = bulk.chrom.len();
    for (what, got) in [("y", bulk.y.len()), ("lambda", bulk.lambda.len())] {
        if got != n {
            return Err(CnvErrors::ShapeMismatch {
                what,
                expected: n,
                got,
            });
        }
    }
    if !(bulk.sigma > 0.0 && bulk.sigma.is_finite()) {
        return Err(CnvErrors::InvalidParameter {
            name: "sigma",
            value: bulk.sigma.to_string(),
            expected: "finite and > 0",
        });
    }
    if !(bulk.d > 0.0 && bulk.d.is_finite()) {
        return Err(CnvErrors::InvalidParameter {
            name: "d",
            value: bulk.d.to_string(),
            expected: "finite and > 0",
        });
    }
    if bulk
        .lambda
        .iter()
        .map(|&l| wide(l))
        .any(|l| l.is_nan() || l <= 0.0)
    {
        return Err(CnvErrors::InvalidParameter {
            name: "lambda",
            value: "a value <= 0".into(),
            expected: "all > 0",
        });
    }
    let log_a = log_transitions(params.t)?;
    let log_d = log_initial(params.t)?;
    let runs = chrom_runs(bulk.chrom)?;
    let paths: Vec<Vec<u8>> = runs
        .par_iter()
        .map(|&(lo, hi)| {
            let e = emission_log_probs(
                &bulk.y[lo..hi],
                &bulk.lambda[lo..hi],
                bulk.d,
                bulk.mu,
                bulk.sigma,
                params.logphi_min,
            );
            viterbi(&log_d, &log_a, &e, N_STATES)
        })
        .collect::<Result<_, _>>()?;
    Ok(paths.concat())
}

/// Decode several pseudobulks in parallel, see [`decode_bulk`].
///
/// ### Params
///
/// * `bulks` - The pseudobulks
/// * `params` - HMM parameters
///
/// ### Returns
///
/// One state path per bulk, or the first input error.
pub fn decode_bulks<T: CnvFloat>(
    bulks: &[BulkHmmInput<T>],
    params: &HmmParams,
) -> Result<Vec<Vec<u8>>, CnvErrors> {
    bulks.par_iter().map(|b| decode_bulk(b, params)).collect()
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
            .unwrap_or_else(|e| panic!("{name}: {e}; run reference/hmm.R"));
        s.lines()
            .skip(1)
            .map(|l| l.split('\t').map(|c| c.trim().to_string()).collect())
            .collect()
    }

    /// Parse an R number, including `-Inf`.
    fn num(s: &str) -> f64 {
        match s {
            "-Inf" => f64::NEG_INFINITY,
            "Inf" => f64::INFINITY,
            _ => s.parse().unwrap(),
        }
    }

    #[test]
    fn test_transitions_and_initial_vs_hahmmr() {
        let rows = read_tsv("hmm_trans.tsv");
        for t in [1e-5, 1e-3] {
            let a = log_transitions(t).unwrap();
            let d = log_initial(t).unwrap();
            let mut seen = 0;
            for r in rows.iter().filter(|r| num(&r[0]) == t) {
                let (from, to): (usize, usize) = (r[1].parse().unwrap(), r[2].parse().unwrap());
                for (ours, theirs) in [(a[from * N_STATES + to], num(&r[3])), (d[to], num(&r[4]))] {
                    if theirs == f64::NEG_INFINITY {
                        assert_eq!(ours, theirs, "t {t} {from}->{to}");
                    } else {
                        assert!(
                            (ours - theirs).abs() <= 1e-12,
                            "t {t} {from}->{to}: {ours} vs {theirs}"
                        );
                    }
                }
                seen += 1;
            }
            assert_eq!(seen, N_STATES * N_STATES);
        }
    }

    #[test]
    fn test_viterbi_paths_vs_hahmmr() {
        let log_a = log_transitions(1e-5).unwrap();
        let log_d = log_initial(1e-5).unwrap();
        for k in 1..=3 {
            let rows = read_tsv(&format!("hmm_viterbi_{k}.tsv"));
            let mut e = Vec::with_capacity(rows.len() * N_STATES);
            let mut z_r = Vec::with_capacity(rows.len());
            for r in &rows {
                e.extend(r[..N_STATES].iter().map(|s| num(s)));
                z_r.push(r[N_STATES].parse::<u8>().unwrap());
            }
            let z = viterbi(&log_d, &log_a, &e, N_STATES).unwrap();
            assert_eq!(z, z_r, "case {k}");
        }
    }

    #[test]
    fn test_decode_bulk_e2e_vs_hahmmr() {
        let meta = &read_tsv("hmm_e2e_meta.tsv")[0];
        let (mu, sigma, d, t, lp) = (
            num(&meta[0]),
            num(&meta[1]),
            num(&meta[2]),
            num(&meta[3]),
            num(&meta[4]),
        );
        let rows = read_tsv("hmm_e2e.tsv");
        let chrom: Vec<u32> = rows.iter().map(|r| r[0].parse().unwrap()).collect();
        let y: Vec<u32> = rows.iter().map(|r| r[1].parse().unwrap()).collect();
        let lambda: Vec<f64> = rows.iter().map(|r| num(&r[2])).collect();
        let z_r: Vec<u8> = rows.iter().map(|r| r[3].parse().unwrap()).collect();
        let bulk = BulkHmmInput {
            chrom: &chrom,
            y: &y,
            lambda: &lambda,
            d,
            mu,
            sigma,
        };
        let params = HmmParams::new(t, lp);
        let z = decode_bulk(&bulk, &params).unwrap();
        assert_eq!(z, z_r);
        // f32 storage decodes the same path here
        let l32: Vec<f32> = lambda.iter().map(|&v| v as f32).collect();
        let b32 = BulkHmmInput {
            chrom: &chrom,
            y: &y,
            lambda: &l32,
            d,
            mu,
            sigma,
        };
        let many = decode_bulks(&[bulk, bulk], &params).unwrap();
        assert_eq!(many, vec![z_r.clone(), z_r.clone()]);
        assert_eq!(decode_bulk(&b32, &params).unwrap(), z_r);
    }

    #[test]
    fn test_viterbi_ties_and_edges() {
        // all-equal scores: lowest index wins at every step
        let m = 3;
        let z = viterbi(&[0.0; 3], &[0.0; 9], &[0.0; 12], m).unwrap();
        assert_eq!(z, vec![0; 4]);
        assert!(viterbi(&[0.0; 3], &[0.0; 9], &[], m).unwrap().is_empty());
        assert!(viterbi(&[0.0; 2], &[0.0; 9], &[0.0; 3], m).is_err());
        assert!(viterbi(&[0.0; 3], &[0.0; 9], &[0.0; 4], m).is_err());
    }

    #[test]
    fn test_chrom_runs_rejects_non_contiguous() {
        assert_eq!(chrom_runs(&[1, 1, 2, 2, 2]).unwrap(), vec![(0, 2), (2, 5)]);
        assert!(chrom_runs(&[1, 2, 1]).is_err());
        assert!(chrom_runs(&[]).unwrap().is_empty());
    }
}
