//! Reference profiles: gene filter, per-cell reference choice, mixture fit (SPEC 2).
//!
//! Everything here works in genomic order (see [`GenomeOrder`]) and in `f64`:
//! profiles are genes x R, small next to the count matrix.

use rayon::prelude::*;

use crate::errors::CnvErrors;
use crate::genome::{CsrCounts, GenomeOrder};
use crate::utils::traits::{CnvFloat, wide};

////////////
// Consts //
////////////

/// Scale from proportions to counts per million, as numbat writes `* 1e6`.
const CPM: f64 = 1e6;

/// A gene counts as expressed above this many counts per million
/// (`min_both`, N/R/utils.R:155; `* 1e6 > 2`, N/R/utils.R:33).
const MIN_CPM: f64 = 2.0;

/// `min_lambda` of `fit_ref_sse` (N/R/utils.R:409).
const FIT_MIN_LAMBDA: f64 = 2e-6;

/// Lower bound on each weight in `fit_ref_sse` (N/R/utils.R:441).
const FIT_LOWER: f64 = 1e-6;

/// Iteration cap of the bounded Levenberg-Marquardt fit. Ours, a safety
/// net; the fit returns its best point when it is hit.
const FIT_MAX_ITER: usize = 500;

/// Stop once an accepted step lowers the objective by less than this
/// fraction. Ours.
const FIT_REL_TOL: f64 = 1e-15;

/// Levenberg-Marquardt damping: start at this fraction of the largest
/// Gauss-Newton diagonal, shrink after an accepted step, grow after a
/// rejected one, give up beyond `LM_MU_MAX` times the diagonal. Textbook
/// values, ours.
const LM_MU_INIT: f64 = 1e-3;

/// See [`LM_MU_INIT`].
const LM_SHRINK: f64 = 0.3;

/// See [`LM_MU_INIT`].
const LM_GROW: f64 = 10.0;

/// See [`LM_MU_INIT`].
const LM_MU_MAX: f64 = 1e20;

///////////////
// Reference //
///////////////

/// How the caller supplies the reference (SPEC 2.1).
#[derive(Clone, Copy, Debug)]
pub enum Reference<'a, T: CnvFloat> {
    /// Normalised profiles, genes x `n_ref`, column-major (each profile
    /// contiguous), caller gene order
    Profiles {
        /// `n_genes * n_ref` proportions, each column sums to 1
        lambda: &'a [T],
        /// Number of profiles
        n_ref: usize,
    },
    /// Indices of normal cells, pooled into one profile
    Cells(&'a [u32]),
}

/// Reference profiles in genomic order.
#[derive(Clone, Debug)]
pub struct Profiles {
    /// Number of profiles `R`
    pub n_ref: usize,
    /// `R * n_genes` proportions, profile-major, genomic order
    pub lambda: Vec<f64>,
}

impl Profiles {
    /// Number of genes.
    ///
    /// ### Returns
    ///
    /// Genes per profile.
    #[inline]
    pub fn n_genes(&self) -> usize {
        self.lambda.len() / self.n_ref
    }

    /// One profile.
    ///
    /// ### Params
    ///
    /// * `r` - Profile index
    ///
    /// ### Returns
    ///
    /// Proportions per gene, genomic order.
    #[inline]
    pub fn profile(&self, r: usize) -> &[f64] {
        let n = self.n_genes();
        &self.lambda[r * n..(r + 1) * n]
    }
}

impl<T: CnvFloat> Reference<'_, T> {
    /// Resolve into genomic-order profiles.
    ///
    /// ### Params
    ///
    /// * `counts` - Count matrix (used for [`Reference::Cells`])
    /// * `order` - Genomic order of the genes
    ///
    /// ### Returns
    ///
    /// The profiles, or an error for a malformed reference.
    pub fn resolve(&self, counts: &CsrCounts, order: &GenomeOrder) -> Result<Profiles, CnvErrors> {
        match *self {
            Reference::Profiles { lambda, n_ref } => profiles_genomic(lambda, n_ref, order),
            Reference::Cells(cells) => Ok(Profiles {
                n_ref: 1,
                lambda: aggregate(counts, order, cells)?,
            }),
        }
    }
}

/// Check caller-order profiles and permute them into genomic order.
///
/// ### Params
///
/// * `lambda` - `n_genes * n_ref` proportions, profile-major, caller order
/// * `n_ref` - Number of profiles
/// * `order` - Genomic order of the genes
///
/// ### Returns
///
/// The profiles, or an error for a malformed reference.
pub fn profiles_genomic<T: CnvFloat>(
    lambda: &[T],
    n_ref: usize,
    order: &GenomeOrder,
) -> Result<Profiles, CnvErrors> {
    let n = order.n_genes();
    if n_ref == 0 {
        return Err(CnvErrors::EmptyInput {
            what: "reference profiles",
        });
    }
    if lambda.len() != n * n_ref {
        return Err(CnvErrors::ShapeMismatch {
            what: "reference profiles",
            expected: n * n_ref,
            got: lambda.len(),
        });
    }
    let mut out = Vec::with_capacity(n * n_ref);
    for col in lambda.chunks_exact(n) {
        out.extend(order.perm.iter().map(|&g| wide(col[g as usize])));
    }
    if out.iter().any(|&x| !(x.is_finite() && x >= 0.0)) {
        return Err(CnvErrors::InvalidReference {
            what: "profiles must be finite and non-negative",
        });
    }
    Ok(Profiles { n_ref, lambda: out })
}

/// Pool cells into one normalised profile (`aggregate_counts`, SPEC 2.1).
///
/// ### Params
///
/// * `counts` - Count matrix
/// * `order` - Genomic order of the genes
/// * `cells` - Cells to pool
///
/// ### Returns
///
/// `lambda_g = sum_c Y_gc / sum_{g,c} Y_gc`, genomic order, or an error if
/// the cells are empty or have no counts.
pub fn aggregate(
    counts: &CsrCounts,
    order: &GenomeOrder,
    cells: &[u32],
) -> Result<Vec<f64>, CnvErrors> {
    check_genes(counts, order)?;
    if cells.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "reference cells",
        });
    }
    profile_from_sums(&order.to_genomic(&counts.gene_sums(Some(cells))?)?)
}

/// Normalise pooled gene sums into a profile (SPEC 2.1).
///
/// ### Params
///
/// * `sums` - Summed counts per gene over the reference cells
///
/// ### Returns
///
/// `sums / sum(sums)`, same order, or an error if every sum is zero.
pub fn profile_from_sums(sums: &[u64]) -> Result<Vec<f64>, CnvErrors> {
    let total: u64 = sums.iter().sum();
    if total == 0 {
        return Err(CnvErrors::InvalidReference {
            what: "reference cells have no counts",
        });
    }
    Ok(sums.iter().map(|&s| s as f64 / total as f64).collect())
}

//////////////////
// filter_genes //
//////////////////

/// Mutually expressed genes (`filter_genes`, SPEC 2.2).
///
/// ### Params
///
/// * `counts` - Count matrix
/// * `order` - Genomic order of the genes
/// * `lambda_ref` - One reference profile, genomic order
/// * `cells` - Cells for `lambda_obs`; `None` for all
/// * `allowed` - From [`GenomeOrder::allowed_mask`]; excluded genes are
///   dropped before `lambda_obs` is normalised
///
/// ### Returns
///
/// Keep mask per gene, genomic order.
pub fn filter_genes(
    counts: &CsrCounts,
    order: &GenomeOrder,
    lambda_ref: &[f64],
    cells: Option<&[u32]>,
    allowed: &[bool],
) -> Result<Vec<bool>, CnvErrors> {
    check_genes(counts, order)?;
    let n = order.n_genes();
    for (what, len) in [("lambda_ref", lambda_ref.len()), ("allowed", allowed.len())] {
        if len != n {
            return Err(CnvErrors::ShapeMismatch {
                what,
                expected: n,
                got: len,
            });
        }
    }
    filter_genes_from_sums(
        &order.to_genomic(&counts.gene_sums(cells)?)?,
        lambda_ref,
        allowed,
    )
}

/// [`filter_genes`] from gene sums already pooled over the cells.
///
/// ### Params
///
/// * `sums` - Summed counts per gene, genomic order
/// * `lambda_ref` - One reference profile, genomic order
/// * `allowed` - From [`GenomeOrder::allowed_mask`]
///
/// ### Returns
///
/// Keep mask per gene, genomic order, or a shape error.
pub fn filter_genes_from_sums(
    sums: &[u64],
    lambda_ref: &[f64],
    allowed: &[bool],
) -> Result<Vec<bool>, CnvErrors> {
    let n = sums.len();
    for (what, len) in [("lambda_ref", lambda_ref.len()), ("allowed", allowed.len())] {
        if len != n {
            return Err(CnvErrors::ShapeMismatch {
                what,
                expected: n,
                got: len,
            });
        }
    }
    let total: u64 = (0..n).filter(|&i| allowed[i]).map(|i| sums[i]).sum();
    if total == 0 {
        return Err(CnvErrors::EmptyInput {
            what: "counts over the allowed genes",
        });
    }
    let obs: Vec<f64> = sums.iter().map(|&s| s as f64 / total as f64).collect();
    let mean_nz = |x: &[f64]| {
        let (s, k) = (0..n)
            .filter(|&i| allowed[i] && x[i] != 0.0)
            .fold((0.0, 0usize), |(s, k), i| (s + x[i], k + 1));
        s / k as f64
    };
    let (ref_mean, obs_mean) = (mean_nz(lambda_ref), mean_nz(&obs));
    Ok((0..n)
        .map(|i| {
            let (r, o) = (lambda_ref[i], obs[i]);
            allowed[i]
                && r > 0.0
                && ((r * CPM > MIN_CPM && o * CPM > MIN_CPM) || r > ref_mean || o > obs_mean)
        })
        .collect())
}

////////////////////
// choose_ref_cor //
////////////////////

/// Per-cell reference by Pearson correlation (`choose_ref_cor`, SPEC 2.3).
///
/// `ln(x*1e6+1)` is zero at a zero count, so each cell's sums run over its
/// nonzeros only; the reference side is centred once. Parallel over cells.
///
/// ### Params
///
/// * `counts` - Count matrix
/// * `order` - Genomic order of the genes
/// * `profiles` - Reference profiles
///
/// ### Returns
///
/// Reference index per cell, `None` for cells with no counts in the selected
/// genes (or a degenerate correlation). With one profile every cell gets 0.
pub fn choose_ref_cor(
    counts: &CsrCounts,
    order: &GenomeOrder,
    profiles: &Profiles,
) -> Result<Vec<Option<u32>>, CnvErrors> {
    check_genes(counts, order)?;
    check_profiles(profiles, order)?;
    let n_ref = profiles.n_ref;
    if n_ref == 1 {
        return Ok(vec![Some(0); counts.n_cells()]);
    }
    let n = order.n_genes();
    // sel[i]: index among selected genes, genomic order; u32::MAX if not selected
    let mut sel = vec![u32::MAX; n];
    let mut n_sel = 0usize;
    for i in 0..n {
        if (0..n_ref).any(|r| profiles.profile(r)[i] * CPM > MIN_CPM) {
            sel[i] = n_sel as u32;
            n_sel += 1;
        }
    }
    if n_sel < 2 {
        return Err(CnvErrors::EmptyInput {
            what: "genes expressed in a reference",
        });
    }
    // centred ln-profiles, gene-major [k * n_ref + r]
    let mut bc = vec![0.0f64; n_sel * n_ref];
    let mut ss_b = vec![0.0f64; n_ref];
    for r in 0..n_ref {
        let p = profiles.profile(r);
        let b: Vec<f64> = (0..n)
            .filter(|&i| sel[i] != u32::MAX)
            .map(|i| (p[i] * CPM).ln_1p())
            .collect();
        let mean = b.iter().sum::<f64>() / n_sel as f64;
        for (k, &v) in b.iter().enumerate() {
            let d = v - mean;
            bc[k * n_ref + r] = d;
            ss_b[r] += d * d;
        }
    }
    // caller gene -> selected index
    let sel_of: Vec<u32> = order.rank.iter().map(|&i| sel[i as usize]).collect();
    let nf = n_sel as f64;
    Ok((0..counts.n_cells())
        .into_par_iter()
        .map_init(
            || vec![0.0f64; n_ref],
            |sab, c| {
                let (idx, val) = counts.row(c);
                let lib: u64 = val.iter().map(|&y| y as u64).sum();
                let (mut sa, mut saa) = (0.0f64, 0.0f64);
                sab.fill(0.0);
                for (&j, &y) in idx.iter().zip(val) {
                    let k = sel_of[j as usize];
                    if k == u32::MAX || y == 0 {
                        continue;
                    }
                    let a = (y as f64 / lib as f64 * CPM).ln_1p();
                    sa += a;
                    saa += a * a;
                    let b = &bc[k as usize * n_ref..(k as usize + 1) * n_ref];
                    for r in 0..n_ref {
                        sab[r] += a * b[r];
                    }
                }
                if sa == 0.0 {
                    return None;
                }
                let var_a = saa - sa * sa / nf;
                let mut best: Option<(u32, f64)> = None;
                for r in 0..n_ref {
                    let rho = sab[r] / (var_a * ss_b[r]).sqrt();
                    if rho.is_finite() && best.is_none_or(|(_, b)| rho > b) {
                        best = Some((r as u32, rho));
                    }
                }
                best.map(|(r, _)| r)
            },
        )
        .collect())
}

/// Reference mixture from per-cell choices (`get_lambdas_bar`, SPEC 2.4).
///
/// ### Params
///
/// * `profiles` - Reference profiles
/// * `refs` - From [`choose_ref_cor`]; `None` cells are ignored
///
/// ### Returns
///
/// `lambda_bar = Lambda w` with `w_r` the share of assigned cells choosing
/// `r`, genomic order.
pub fn lambdas_bar(profiles: &Profiles, refs: &[Option<u32>]) -> Result<Vec<f64>, CnvErrors> {
    let mut w = vec![0.0f64; profiles.n_ref];
    for &r in refs.iter().flatten() {
        let r = r as usize;
        if r >= profiles.n_ref {
            return Err(CnvErrors::InvalidParameter {
                name: "refs",
                value: r.to_string(),
                expected: "reference indices below n_ref",
            });
        }
        w[r] += 1.0;
    }
    let total: f64 = w.iter().sum();
    if total == 0.0 {
        return Err(CnvErrors::EmptyInput {
            what: "cells with a reference",
        });
    }
    w.iter_mut().for_each(|x| *x /= total);
    Ok(mix(profiles, &w))
}

/////////////////
// fit_ref_sse //
/////////////////

/// Result of [`fit_ref_sse`].
#[derive(Clone, Debug)]
pub struct RefFit {
    /// Mixture weights, summing to 1
    pub w: Vec<f64>,
    /// `Lambda w`, genomic order
    pub lambda_bar: Vec<f64>,
    /// Objective over genes used; `None` for a single profile, as numbat
    pub mse: Option<f64>,
}

/// Fit a reference mixture to a pseudobulk (`fit_ref_sse`, SPEC 2.5).
///
/// Minimises `f(w) = sum_g [ln(lambda_obs_g / (Lambda w / sum w)_g)]^2`
/// subject to `w_r >= 1e-6`, by bounded Levenberg-Marquardt on the
/// residuals with the analytic Jacobian. numbat uses L-BFGS-B with a
/// numerical gradient; parity is on the final objective.
///
/// ### Params
///
/// * `y` - Pseudobulk counts per gene, genomic order
/// * `profiles` - Reference profiles
///
/// ### Returns
///
/// Weights, `lambda_bar` and `mse`, or an error if no gene qualifies.
pub fn fit_ref_sse(y: &[u64], profiles: &Profiles) -> Result<RefFit, CnvErrors> {
    let n = profiles.n_genes();
    if y.len() != n {
        return Err(CnvErrors::ShapeMismatch {
            what: "pseudobulk counts",
            expected: n,
            got: y.len(),
        });
    }
    let n_ref = profiles.n_ref;
    if n_ref == 1 {
        return Ok(RefFit {
            w: vec![1.0],
            lambda_bar: profiles.lambda.clone(),
            mse: None,
        });
    }
    let genes: Vec<usize> = (0..n)
        .filter(|&i| {
            y[i] > 0
                && (0..n_ref).map(|r| profiles.profile(r)[i]).sum::<f64>() / (n_ref as f64)
                    > FIT_MIN_LAMBDA
        })
        .collect();
    if genes.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "genes for fit_ref_sse",
        });
    }
    let total: u64 = genes.iter().map(|&i| y[i]).sum();
    let ln_obs: Vec<f64> = genes
        .iter()
        .map(|&i| (y[i] as f64 / total as f64).ln())
        .collect();
    // gene-major [k * n_ref + r]
    let lam: Vec<f64> = genes
        .iter()
        .flat_map(|&i| (0..n_ref).map(move |r| profiles.profile(r)[i]))
        .collect();
    let (w, f) = fit_bounded_lm(&ln_obs, &lam, n_ref);
    let s: f64 = w.iter().sum();
    let w: Vec<f64> = w.iter().map(|x| x / s).collect();
    Ok(RefFit {
        lambda_bar: mix(profiles, &w),
        w,
        mse: Some(f / genes.len() as f64),
    })
}

/// Residuals `e_g = ln_obs_g - ln(Lambda w)_g + ln(sum w)` and the objective.
///
/// ### Params
///
/// * `ln_obs` - `ln(lambda_obs)` per gene
/// * `lam` - Profiles, gene-major
/// * `w` - Weights
/// * `e` - Residuals, written
/// * `m` - `(Lambda w)_g`, written
///
/// ### Returns
///
/// `sum_g e_g^2`.
fn residuals(ln_obs: &[f64], lam: &[f64], w: &[f64], e: &mut [f64], m: &mut [f64]) -> f64 {
    let n_ref = w.len();
    let ln_s = w.iter().sum::<f64>().ln();
    let mut f = 0.0;
    for (k, row) in lam.chunks_exact(n_ref).enumerate() {
        m[k] = row.iter().zip(w).map(|(a, b)| a * b).sum();
        e[k] = ln_obs[k] - m[k].ln() + ln_s;
        f += e[k] * e[k];
    }
    f
}

/// Bounded Levenberg-Marquardt for [`fit_ref_sse`].
///
/// Jacobian `J_gr = 1/sum(w) - Lambda_gr / (Lambda w)_g`. `J w = 0` (the
/// objective is scale-free), so the Gauss-Newton matrix is singular and the
/// damping does the regularising. Variables at the bound with a positive
/// gradient are held fixed; the step is projected onto the bound.
///
/// ### Params
///
/// * `ln_obs` - `ln(lambda_obs)` per gene
/// * `lam` - Profiles, gene-major
/// * `n_ref` - Number of profiles
///
/// ### Returns
///
/// `(w, f)`: unnormalised weights and the final objective.
fn fit_bounded_lm(ln_obs: &[f64], lam: &[f64], n_ref: usize) -> (Vec<f64>, f64) {
    let ng = ln_obs.len();
    let mut w = vec![1.0 / n_ref as f64; n_ref];
    let (mut e, mut m) = (vec![0.0; ng], vec![0.0; ng]);
    let (mut e_new, mut m_new) = (vec![0.0; ng], vec![0.0; ng]);
    let mut f = residuals(ln_obs, lam, &w, &mut e, &mut m);
    let mut mu = 0.0;
    let mut g = vec![0.0; n_ref];
    let mut h = vec![0.0; n_ref * n_ref];
    let mut jr = vec![0.0; n_ref];
    for _ in 0..FIT_MAX_ITER {
        let inv_s = 1.0 / w.iter().sum::<f64>();
        g.fill(0.0);
        h.fill(0.0);
        for k in 0..ng {
            let row = &lam[k * n_ref..(k + 1) * n_ref];
            for r in 0..n_ref {
                jr[r] = inv_s - row[r] / m[k];
            }
            for r in 0..n_ref {
                g[r] += 2.0 * jr[r] * e[k];
                for q in 0..n_ref {
                    h[r * n_ref + q] += 2.0 * jr[r] * jr[q];
                }
            }
        }
        let free: Vec<usize> = (0..n_ref)
            .filter(|&r| !(w[r] <= FIT_LOWER && g[r] > 0.0))
            .collect();
        if free.is_empty() {
            break;
        }
        let max_diag = free.iter().map(|&r| h[r * n_ref + r]).fold(0.0, f64::max);
        if mu == 0.0 {
            mu = LM_MU_INIT * max_diag.max(f64::MIN_POSITIVE);
        }
        let nf = free.len();
        let mut accepted = false;
        while !accepted && mu < LM_MU_MAX * max_diag.max(1.0) {
            let mut a = vec![0.0; nf * nf];
            let mut b = vec![0.0; nf];
            for (i, &r) in free.iter().enumerate() {
                b[i] = -g[r];
                for (j, &q) in free.iter().enumerate() {
                    a[i * nf + j] = h[r * n_ref + q];
                }
                a[i * nf + i] += mu;
            }
            if !solve_spd(&mut a, &mut b, nf) {
                mu *= LM_GROW;
                continue;
            }
            let mut w_new = w.clone();
            for (i, &r) in free.iter().enumerate() {
                w_new[r] = (w[r] + b[i]).max(FIT_LOWER);
            }
            let f_new = residuals(ln_obs, lam, &w_new, &mut e_new, &mut m_new);
            if f_new < f {
                accepted = true;
                let done = f - f_new <= FIT_REL_TOL * f;
                w = w_new;
                f = f_new;
                std::mem::swap(&mut e, &mut e_new);
                std::mem::swap(&mut m, &mut m_new);
                mu *= LM_SHRINK;
                if done {
                    return (w, f);
                }
            } else {
                mu *= LM_GROW;
            }
        }
        if !accepted {
            break;
        }
    }
    (w, f)
}

/// Solve `A x = b` in place for small symmetric positive definite `A`
/// (Cholesky).
///
/// ### Params
///
/// * `a` - Row-major `n x n`, overwritten with the factor
/// * `b` - Right-hand side, overwritten with `x`
/// * `n` - Dimension
///
/// ### Returns
///
/// `false` if `A` is not numerically positive definite.
fn solve_spd(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d.is_nan() || d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
    }
    for i in 0..n {
        for k in 0..i {
            b[i] -= a[i * n + k] * b[k];
        }
        b[i] /= a[i * n + i];
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            b[i] -= a[k * n + i] * b[k];
        }
        b[i] /= a[i * n + i];
    }
    true
}

/// `Lambda w` over all genes.
///
/// ### Params
///
/// * `profiles` - Reference profiles
/// * `w` - Weights, one per profile
///
/// ### Returns
///
/// Mixed profile, genomic order.
fn mix(profiles: &Profiles, w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; profiles.n_genes()];
    for (r, &wr) in w.iter().enumerate() {
        out.iter_mut()
            .zip(profiles.profile(r))
            .for_each(|(o, &l)| *o += wr * l);
    }
    out
}

/// Check that the count matrix and the genome agree on the gene count.
///
/// ### Params
///
/// * `counts` - Count matrix
/// * `order` - Genomic order
///
/// ### Returns
///
/// `Ok` or a shape error.
fn check_genes(counts: &CsrCounts, order: &GenomeOrder) -> Result<(), CnvErrors> {
    if counts.n_genes != order.n_genes() {
        return Err(CnvErrors::ShapeMismatch {
            what: "genes in counts vs genome",
            expected: order.n_genes(),
            got: counts.n_genes,
        });
    }
    Ok(())
}

/// Check that profiles cover the genome.
///
/// ### Params
///
/// * `profiles` - Reference profiles
/// * `order` - Genomic order
///
/// ### Returns
///
/// `Ok` or a shape error.
fn check_profiles(profiles: &Profiles, order: &GenomeOrder) -> Result<(), CnvErrors> {
    if profiles.n_ref == 0 || profiles.lambda.len() != profiles.n_ref * order.n_genes() {
        return Err(CnvErrors::ShapeMismatch {
            what: "reference profiles",
            expected: profiles.n_ref.max(1) * order.n_genes(),
            got: profiles.lambda.len(),
        });
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixtures");

    /// Rows of a TSV fixture, header dropped.
    pub(crate) fn read_tsv(name: &str) -> Vec<Vec<String>> {
        let s = std::fs::read_to_string(format!("{FIXTURES}/{name}"))
            .unwrap_or_else(|e| panic!("{name}: {e}; run reference/reference.R"));
        s.lines()
            .skip(1)
            .map(|l| l.split('\t').map(|c| c.trim().to_string()).collect())
            .collect()
    }

    /// The simulated data of reference/reference.R.
    pub(crate) struct Sim {
        /// Genomic order
        pub order: GenomeOrder,
        /// CSR row offsets
        pub indptr: Vec<u32>,
        /// CSR gene indices
        pub indices: Vec<u32>,
        /// CSR counts
        pub data: Vec<u32>,
        /// Profiles, column-major, caller order
        pub lambda: Vec<f64>,
    }

    impl Sim {
        /// Borrow the counts.
        pub fn counts(&self) -> CsrCounts<'_> {
            CsrCounts::new(
                &self.indptr,
                &self.indices,
                &self.data,
                self.order.n_genes(),
            )
            .unwrap()
        }
    }

    /// numbat's MHC interval (N/R/utils.R:147-149).
    pub(crate) const MHC: (u32, u64, u64) = (6, 28_510_120, 33_480_577);

    /// Load the simulated data.
    pub(crate) fn load_sim() -> Sim {
        let genes = read_tsv("ref_genes.tsv");
        let col = |j: usize| -> Vec<u64> { genes.iter().map(|r| r[j].parse().unwrap()).collect() };
        let chrom: Vec<u32> = col(1).into_iter().map(|x| x as u32).collect();
        let order = GenomeOrder::new(&chrom, &col(2), &col(3)).unwrap();
        let n = order.n_genes();
        let trip: Vec<(usize, u32, u32)> = read_tsv("ref_counts.tsv")
            .iter()
            .map(|r| {
                (
                    r[0].parse().unwrap(),
                    r[1].parse().unwrap(),
                    r[2].parse().unwrap(),
                )
            })
            .collect();
        let n_cells = trip.iter().map(|t| t.0).max().unwrap() + 1;
        let mut rows: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_cells];
        for (c, g, y) in trip {
            rows[c].push((g, y));
        }
        let mut indptr = vec![0u32];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for row in rows {
            for (g, y) in row {
                indices.push(g);
                data.push(y);
            }
            indptr.push(indices.len() as u32);
        }
        let prof = read_tsv("ref_profiles.tsv");
        let mut lambda = vec![0.0; 3 * n];
        for (g, r) in prof.iter().enumerate() {
            for k in 0..3 {
                lambda[k * n + g] = r[k + 1].parse().unwrap();
            }
        }
        Sim {
            order,
            indptr,
            indices,
            data,
            lambda,
        }
    }

    /// Caller-order column of `ref_lambdas.tsv` in genomic order.
    pub(crate) fn lambdas_col(order: &GenomeOrder, j: usize) -> Vec<f64> {
        let v: Vec<f64> = read_tsv("ref_lambdas.tsv")
            .iter()
            .map(|r| r[j].parse().unwrap())
            .collect();
        order.to_genomic(&v).unwrap()
    }

    /// Kept set from R as a genomic-order mask.
    fn r_mask(order: &GenomeOrder, rows: &[Vec<String>], set: &str) -> Vec<bool> {
        let mut m = vec![false; order.n_genes()];
        for r in rows.iter().filter(|r| r[0] == set) {
            m[order.rank[r[1].parse::<usize>().unwrap()] as usize] = true;
        }
        m
    }

    #[test]
    fn test_choose_ref_cor_matches_numbat() {
        let sim = load_sim();
        let counts = sim.counts();
        let p = Reference::Profiles {
            lambda: &sim.lambda,
            n_ref: 3,
        }
        .resolve(&counts, &sim.order)
        .unwrap();
        let refs = choose_ref_cor(&counts, &sim.order, &p).unwrap();
        let want: Vec<Option<u32>> = read_tsv("ref_choose.tsv")
            .iter()
            .map(|r| r[1].parse().ok())
            .collect();
        assert_eq!(refs, want);
        let bar = lambdas_bar(&p, &refs).unwrap();
        for (a, b) in bar.iter().zip(lambdas_col(&sim.order, 1)) {
            assert!((a - b).abs() <= 1e-13 * b.max(1e-300), "{a} vs {b}");
        }
    }

    #[test]
    fn test_choose_ref_cor_single_profile_and_zero_cell() {
        let order = GenomeOrder::new(&[1, 1, 1], &[1, 2, 3], &[1, 2, 3]).unwrap();
        let counts = CsrCounts::new(&[0, 2, 2], &[0, 2], &[3, 1], 3).unwrap();
        let one = Profiles {
            n_ref: 1,
            lambda: vec![0.5, 0.25, 0.25],
        };
        assert_eq!(
            choose_ref_cor(&counts, &order, &one).unwrap(),
            vec![Some(0); 2]
        );
        let two = Profiles {
            n_ref: 2,
            lambda: vec![0.5, 0.25, 0.25, 0.25, 0.25, 0.5],
        };
        let r = choose_ref_cor(&counts, &order, &two).unwrap();
        assert_eq!(r[1], None);
    }

    #[test]
    fn test_filter_genes_and_aggregate_match_numbat() {
        let sim = load_sim();
        let counts = sim.counts();
        let o = &sim.order;
        let allowed = o.allowed_mask(&[MHC]);
        let rows = read_tsv("ref_filter.tsv");
        let bar = lambdas_col(o, 1);
        let all = filter_genes(&counts, o, &bar, None, &allowed).unwrap();
        assert_eq!(all, r_mask(o, &rows, "all"));
        let sub: Vec<u32> = (150..300).collect();
        let s = filter_genes(&counts, o, &bar, Some(&sub), &allowed).unwrap();
        assert_eq!(s, r_mask(o, &rows, "subset"));

        let cells: Vec<u32> = (0..100).collect();
        let agg = Reference::<f64>::Cells(&cells).resolve(&counts, o).unwrap();
        for (a, b) in agg.lambda.iter().zip(lambdas_col(o, 2)) {
            assert!((a - b).abs() <= 1e-13 * b.max(1e-300), "{a} vs {b}");
        }
        let a = filter_genes(&counts, o, &agg.lambda, None, &allowed).unwrap();
        assert_eq!(a, r_mask(o, &rows, "aggregate"));
    }

    #[test]
    fn test_fit_ref_sse_objective_not_worse_than_numbat() {
        let sim = load_sim();
        let counts = sim.counts();
        let o = &sim.order;
        let n = o.n_genes();
        for row in read_tsv("ref_fit.tsv") {
            let cells: Vec<u32> = row[1].split(',').map(|x| x.parse().unwrap()).collect();
            let refs: Vec<usize> = row[2].split(',').map(|x| x.parse().unwrap()).collect();
            let lambda: Vec<f64> = refs
                .iter()
                .flat_map(|&r| sim.lambda[r * n..(r + 1) * n].iter().copied())
                .collect();
            let p = Reference::Profiles {
                lambda: &lambda,
                n_ref: refs.len(),
            }
            .resolve(&counts, o)
            .unwrap();
            let y = o
                .to_genomic(&counts.gene_sums(Some(&cells)).unwrap())
                .unwrap();
            let fit = fit_ref_sse(&y, &p).unwrap();
            let r_mse: f64 = row[6].parse().unwrap();
            let mse = fit.mse.unwrap();
            let r_w: Vec<f64> = refs.iter().map(|&r| row[3 + r].parse().unwrap()).collect();
            println!(
                "{}: mse {mse:.17e} R {r_mse:.17e} w {:?} R {:?}",
                row[0], fit.w, r_w
            );
            assert!(mse <= r_mse * (1.0 + 1e-6), "{}: {mse} > {r_mse}", row[0]);
            assert!((fit.w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
}
