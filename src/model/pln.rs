//! Poisson-lognormal emission (SPEC 4).
//!
//! The density is a Laplace-centred Gauss-Hermite quadrature: Newton for the
//! mode of the log-integrand, then a fixed node set scaled by the curvature at
//! the mode. The same nodes give the posterior moments of the latent log-rate,
//! hence the derivatives in `(mu, sigma)` at no extra transcendental cost.
//!
//! Everything here is sequential on purpose: the callers (per-cell posteriors,
//! pseudobulk fits) parallelise over cells or segments.

use std::f64::consts::PI;
use std::sync::OnceLock;

use crate::errors::CnvErrors;
use crate::utils::optim::maximise_concave_1d;
use crate::utils::traits::{CnvFloat, wide};

////////////
// Consts //
////////////

/// Gauss-Hermite node count for the PLN density (SPEC 4.1). The smallest
/// count under 1e-6 relative error; see `docs/PERFORMANCE.md`.
pub const GH_NODES: usize = 56;

/// Lower bound on `sigma` in [`fit_lnpois`] (H/src/mle.cpp:45).
pub const SIGMA_MIN: f64 = 0.01;

/// Bounds of `phi` for [`phi_mle`] (N/R/utils.R:1425).
pub const PHI_MLE_BOUNDS: (f64, f64) = (0.1, 10.0);

/// Bounds of `phi` for [`approx_phi_post`] (N/R/utils.R:1508).
pub const PHI_POST_BOUNDS: (f64, f64) = (0.2, 10.0);

/// Newton iteration cap for the mode of the log-integrand. Convergence is
/// monotone from the starting point; far starts move about one unit of `z`
/// per step, so 200 covers any realistic `mu`.
const MODE_MAX_ITER: usize = 200;

/// Relative step tolerance for the mode. The quadrature is exact under any
/// centre, so the mode only needs to be close enough to keep the nodes on the
/// mass.
const MODE_TOL: f64 = 1e-10;

/// Iteration cap for [`fit_lnpois`]. Our choice; typical fits take under 15.
const FIT_MAX_ITER: usize = 200;

/// Upper clamp on the moment start's `sigma`: one outlying count can inflate
/// the pooled variance. Our choice.
const MOMENT_SIGMA_MAX: f64 = 3.0;

/// Step halvings per line search in [`fit_lnpois`]. Our choice.
const FIT_MAX_HALVINGS: usize = 60;

/// Relative tolerance on the predicted increase `g . d` in [`fit_lnpois`].
/// Our choice: well below the quadrature error of a single gene.
const FIT_TOL: f64 = 1e-13;

/// Relative tolerance on `ln phi` for the phi optimisers. Our choice.
const PHI_TOL: f64 = 1e-12;

/// Iteration cap for the phi optimisers. Our choice.
const PHI_MAX_ITER: usize = 200;

/// Table size for exact `ln(x!)`; Stirling beyond.
const LN_FACT_TABLE: usize = 64;

/////////////
// Helpers //
/////////////

/// Gauss-Hermite nodes and weights pre-scaled for the Laplace substitution.
struct GhTable {
    /// `sqrt(2) t_i` for the Hermite nodes `t_i`
    z: [f64; GH_NODES],
    /// `sqrt(2) w_i exp(t_i^2)`
    w: [f64; GH_NODES],
}

/// Gauss-Hermite rule for weight `exp(-t^2)`, Newton on the orthonormal
/// Hermite recurrence.
///
/// ### Params
///
/// * `n` - Node count
///
/// ### Returns
///
/// `(nodes, weights)`, nodes in descending order.
///
/// ### References
///
/// Press et al., *Numerical Recipes*, 3rd ed., section 4.6 (`gauher`).
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    const PIM4: f64 = 0.751_125_544_464_942_5; // pi^(-1/4)
    let nf = n as f64;
    let mut t = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut z = 0.0f64;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-0.16667),
            1 => z - 1.14 * nf.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * t[0],
            3 => 1.91 * z - 0.91 * t[1],
            _ => 2.0 * z - t[i - 2],
        };
        let mut pp = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (PIM4, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / (jf + 1.0)).sqrt() * p2 - (jf / (jf + 1.0)).sqrt() * p3;
            }
            pp = (2.0 * nf).sqrt() * p2;
            let step = p1 / pp;
            z -= step;
            if step.abs() <= 1e-15 * (1.0 + z.abs()) {
                break;
            }
        }
        t[i] = z;
        t[n - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[n - 1 - i] = w[i];
    }
    (t, w)
}

/// The [`GH_NODES`]-point table, built once.
///
/// ### Returns
///
/// Reference to the shared table.
fn gh_table() -> &'static GhTable {
    static TABLE: OnceLock<GhTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let (t, w) = gauss_hermite(GH_NODES);
        let mut table = GhTable {
            z: [0.0; GH_NODES],
            w: [0.0; GH_NODES],
        };
        for i in 0..GH_NODES {
            table.z[i] = std::f64::consts::SQRT_2 * t[i];
            table.w[i] = std::f64::consts::SQRT_2 * w[i] * (t[i] * t[i]).exp();
        }
        table
    })
}

/// `ln(x!)`: a table for small `x`, Stirling's series beyond.
///
/// ### Params
///
/// * `x` - Count
///
/// ### Returns
///
/// `lgamma(x + 1)`. Absolute error below 1e-13 for every `u32`.
#[inline]
fn ln_factorial(x: u32) -> f64 {
    static TABLE: OnceLock<[f64; LN_FACT_TABLE]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0.0; LN_FACT_TABLE];
        for k in 1..LN_FACT_TABLE {
            t[k] = t[k - 1] + (k as f64).ln();
        }
        t
    });
    if (x as usize) < LN_FACT_TABLE {
        return table[x as usize];
    }
    let n = x as f64;
    let r = 1.0 / n;
    let r2 = r * r;
    n * n.ln() - n
        + 0.5 * (2.0 * PI * n).ln()
        + r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 / 1680.0)))
}

/// Mode of `h(z) = z x - e^z - (z - mu)^2 / (2 sigma^2)`.
///
/// `h'` is concave and decreasing, so Newton started where `h' <= 0`
/// (`z0 = max(mu, ln x)`) converges monotonically from the right.
///
/// ### Params
///
/// * `x` - Count as `f64`
/// * `mu` - Prior mean of `z`
/// * `inv_var` - `1 / sigma^2`
///
/// ### Returns
///
/// `(m, e^m)`.
#[inline]
fn pln_mode(x: f64, mu: f64, inv_var: f64) -> (f64, f64) {
    let mut z = if x > 0.0 { mu.max(x.ln()) } else { mu };
    for _ in 0..MODE_MAX_ITER {
        let ez = z.exp();
        let step = (x - ez - (z - mu) * inv_var) / (-ez - inv_var);
        z -= step;
        if step.abs() <= MODE_TOL * (1.0 + z.abs()) {
            break;
        }
    }
    (z, z.exp())
}

/// Quadrature state shared by [`ln_pln`] and [`ln_pln_fused`].
///
/// ### Params
///
/// * `x` - Count
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
/// * `terms` - Receives the unnormalised node masses
///
/// ### Returns
///
/// `(ln p, sum of terms, s, m - mu)`.
#[inline(always)]
fn pln_quadrature(
    x: u32,
    mu: f64,
    sigma: f64,
    terms: &mut [f64; GH_NODES],
) -> (f64, f64, f64, f64) {
    let gh = gh_table();
    let xf = x as f64;
    let inv_var = 1.0 / (sigma * sigma);
    let (m, em) = pln_mode(xf, mu, inv_var);
    let s = 1.0 / (em + inv_var).sqrt();
    let c = m - mu;
    let half_iv = 0.5 * inv_var;
    let mut sum = 0.0;
    for i in 0..GH_NODES {
        let v = s * gh.z[i];
        // h(m + v) - h(m), written to avoid cancellation
        let dh = xf * v - em * v.exp_m1() - v * (v + 2.0 * c) * half_iv;
        terms[i] = gh.w[i] * dh.exp();
        sum += terms[i];
    }
    let hm = xf * m - em - c * c * half_iv;
    let ln_p = hm - ln_factorial(x) - sigma.ln() - 0.5 * (2.0 * PI).ln() + s.ln() + sum.ln();
    (ln_p, sum, s, c)
}

/////////////
// Density //
/////////////

/// Log-density of the Poisson-lognormal (SPEC 4.1).
///
/// `ln PLN(x; mu, sigma)`, exact (includes `lgamma(x + 1)`). Computed in log
/// space, so it never underflows and needs no floor (**Deviation D9**).
///
/// ### Params
///
/// * `x` - Count
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate; must be `> 0`
///
/// ### Returns
///
/// `ln p`.
#[inline]
pub fn ln_pln(x: u32, mu: f64, sigma: f64) -> f64 {
    let mut terms = [0.0; GH_NODES];
    pln_quadrature(x, mu, sigma, &mut terms).0
}

/// Log-density with first and second derivatives in `(mu, sigma)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlnDerivs {
    /// `ln p`, as in [`ln_pln`]
    pub ln_p: f64,
    /// `d ln p / d mu`
    pub d_mu: f64,
    /// `d ln p / d sigma`
    pub d_sigma: f64,
    /// `d^2 ln p / d mu^2`
    pub d2_mu: f64,
    /// `d^2 ln p / d mu d sigma`
    pub d2_mu_sigma: f64,
    /// `d^2 ln p / d sigma^2`
    pub d2_sigma: f64,
}

impl PlnDerivs {
    /// Element-wise sum, for accumulating over genes.
    ///
    /// ### Params
    ///
    /// * `o` - Term to add
    #[inline]
    fn add(&mut self, o: &PlnDerivs) {
        self.ln_p += o.ln_p;
        self.d_mu += o.d_mu;
        self.d_sigma += o.d_sigma;
        self.d2_mu += o.d2_mu;
        self.d2_mu_sigma += o.d2_mu_sigma;
        self.d2_sigma += o.d2_sigma;
    }
}

/// Log-density and its derivatives in `(mu, sigma)` from one quadrature pass
/// (SPEC 4.3).
///
/// With `u = z - mu` under the posterior of `z` given `x`, mean `a` and
/// central moments `c2, c3, c4`:
/// `d/dmu = a / sigma^2`, `d/dsigma = (c2 + a^2) / sigma^3 - 1 / sigma`, and
/// the second derivatives from `E[d2 ln q] + Cov(d ln q)` of the normal prior
/// `q`. Central moments avoid the `E[u^2] - E[u]^2` cancellation.
///
/// ### Params
///
/// * `x` - Count
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate; must be `> 0`
///
/// ### Returns
///
/// [`PlnDerivs`].
#[inline]
pub fn ln_pln_fused(x: u32, mu: f64, sigma: f64) -> PlnDerivs {
    let mut terms = [0.0; GH_NODES];
    let (ln_p, sum, s, c) = pln_quadrature(x, mu, sigma, &mut terms);
    let gh = gh_table();
    let inv_sum = 1.0 / sum;
    let mut mean_v = 0.0;
    for i in 0..GH_NODES {
        mean_v += terms[i] * gh.z[i];
    }
    mean_v *= s * inv_sum;
    let (mut c2, mut c3, mut c4) = (0.0, 0.0, 0.0);
    for i in 0..GH_NODES {
        let d = s * gh.z[i] - mean_v;
        let d2 = d * d;
        c2 += terms[i] * d2;
        c3 += terms[i] * d2 * d;
        c4 += terms[i] * d2 * d2;
    }
    c2 *= inv_sum;
    c3 *= inv_sum;
    c4 *= inv_sum;

    let a = c + mean_v;
    let iv = 1.0 / (sigma * sigma);
    let is = 1.0 / sigma;
    let eu2 = c2 + a * a;
    PlnDerivs {
        ln_p,
        d_mu: a * iv,
        d_sigma: eu2 * iv * is - is,
        d2_mu: -iv + c2 * iv * iv,
        d2_mu_sigma: -2.0 * a * iv * is + (c3 + 2.0 * a * c2) * iv * iv * is,
        d2_sigma: -3.0 * eu2 * iv * iv
            + iv
            + (c4 + 4.0 * a * c3 + 4.0 * a * a * c2 - c2 * c2) * iv * iv * iv,
    }
}

///////////////
// Gene sets //
///////////////

/// Check the shared inputs of the gene-set functions and build the per-gene
/// offsets `ln(d lambda_g)`.
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, same length, all `> 0`
/// * `d` - Library size, `> 0`
/// * `sigma` - Must be finite and `> 0`, or `None` to skip the check
///
/// ### Returns
///
/// The offsets, or an input error.
fn offsets<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    sigma: Option<f64>,
) -> Result<Vec<f64>, CnvErrors> {
    if lambda.len() != y.len() {
        return Err(CnvErrors::ShapeMismatch {
            what: "lambda vs y",
            expected: y.len(),
            got: lambda.len(),
        });
    }
    if !(d > 0.0 && d.is_finite()) {
        return Err(CnvErrors::InvalidParameter {
            name: "d",
            value: d.to_string(),
            expected: "finite and > 0",
        });
    }
    if let Some(s) = sigma {
        check_sigma(s)?;
    }
    lambda
        .iter()
        .map(|&l| {
            let l = wide(l);
            if l > 0.0 && l.is_finite() {
                Ok((d * l).ln())
            } else {
                Err(CnvErrors::InvalidParameter {
                    name: "lambda",
                    value: l.to_string(),
                    expected: "finite and > 0 (drop genes with lambda = 0 first)",
                })
            }
        })
        .collect()
}

/// Validate `sigma`.
///
/// ### Params
///
/// * `sigma` - Standard deviation of the latent log-rate
///
/// ### Returns
///
/// `Ok` if finite and `> 0`.
fn check_sigma(sigma: f64) -> Result<(), CnvErrors> {
    if sigma > 0.0 && sigma.is_finite() {
        Ok(())
    } else {
        Err(CnvErrors::InvalidParameter {
            name: "sigma",
            value: sigma.to_string(),
            expected: "finite and > 0",
        })
    }
}

/// Segment loglikelihood (SPEC 4.2).
///
/// `sum_g ln PLN(y_g; mu + ln(phi d lambda_g), sigma)`, via
/// [`l_lnpois_multi`]: swept genes first, then the tabled count groups.
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, same length as `y`, all `> 0`
/// * `d` - Library size
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
/// * `phi` - Fold change, `> 0`
///
/// ### Returns
///
/// The loglikelihood, or an input error.
pub fn l_lnpois<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    mu: f64,
    sigma: f64,
    phi: f64,
) -> Result<f64, CnvErrors> {
    let l = l_lnpois_multi(y, lambda, d, mu, sigma, &[phi])?;
    Ok(l[0])
}

/// Segment loglikelihood at several `phi` in one pass over the genes
/// (SPEC 4.2, 8.4).
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, same length as `y`, all `> 0`
/// * `d` - Library size
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
/// * `phis` - Fold changes, each `> 0`
///
/// ### Returns
///
/// One loglikelihood per entry of `phis`, or an input error.
pub fn l_lnpois_multi<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    mu: f64,
    sigma: f64,
    phis: &[f64],
) -> Result<Vec<f64>, CnvErrors> {
    check_sigma(sigma)?;
    let mut shifts = Vec::with_capacity(phis.len());
    for &phi in phis {
        if !(phi > 0.0 && phi.is_finite()) {
            return Err(CnvErrors::InvalidParameter {
                name: "phi",
                value: phi.to_string(),
                expected: "finite and > 0",
            });
        }
        shifts.push(mu + phi.ln());
    }
    let off = offsets(y, lambda, d, None)?;
    // one interpolant per count covers every shift
    let (s_lo, s_hi) = min_max(&shifts);
    let g = CountGroups::new(y, &off, phis.len(), s_hi - s_lo);
    let mut acc = vec![0.0; phis.len()];
    for (&yg, &og) in g.y.iter().zip(&g.off) {
        for (a, &sh) in acc.iter_mut().zip(&shifts) {
            *a += ln_pln(yg, sh + og, sigma);
        }
    }
    for (x, offs) in &g.tabled {
        let (lo, hi) = min_max(offs);
        let it = FusedInterp::new(*x, sigma, s_lo + lo, s_hi + hi);
        for (a, &sh) in acc.iter_mut().zip(&shifts) {
            let mut s = [0.0; N_FORMS];
            for &o in offs {
                it.add_to(sh + o, &mut s);
            }
            *a += s[0];
        }
    }
    Ok(acc)
}

/// Sum of [`ln_pln_fused`] over genes at a common `(mu, sigma)`.
///
/// ### Params
///
/// * `y` - Counts
/// * `off` - Per-gene offsets added to `mu`
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
///
/// ### Returns
///
/// Summed loglikelihood, gradient and Hessian.
fn sum_fused(y: &[u32], off: &[f64], mu: f64, sigma: f64) -> PlnDerivs {
    let mut acc = PlnDerivs::default();
    for (&yg, &og) in y.iter().zip(off) {
        acc.add(&ln_pln_fused(yg, mu + og, sigma));
    }
    acc
}

////////////////////////
// Count interpolants //
////////////////////////

////////////
// Consts //
////////////

/// Largest count whose genes are summed through an interpolant in
/// [`CountGroups`]. Counts above it are swept with [`ln_pln_fused`]. Three is
/// the largest count `test_fused_interp_matches_quadrature` covers.
const MAX_TABLED_COUNT: u32 = 3;

/// Smallest group worth an interpolant: below it the node evaluations cost
/// more than the sweep. Our choice, about four panels of nodes.
const MIN_TABLE_GENES: usize = 64;

/// Chebyshev degree per panel. Chosen with [`CHEB_WIDTH`] so the
/// interpolation error stays far under the quadrature's; see
/// `docs/PERFORMANCE.md`.
const CHEB_DEGREE: usize = 12;

/// Panel width in `t = mu + off`.
const CHEB_WIDTH: f64 = 2.0;

/// Fused terms per node: `ln p` and the five derivatives of [`PlnDerivs`].
const N_FORMS: usize = 6;

/////////////////
// CountGroups //
/////////////////

/// Genes of one fit split by count, for [`sum_grouped`].
///
/// At a fixed count `x` and `sigma`, every term of [`ln_pln_fused`] depends on
/// the gene only through `t = mu + off_g`. A large group of equal counts is
/// therefore summed by evaluating the quadrature at the Chebyshev nodes of
/// panels covering its `t` range, then interpolating every gene, rather than
/// running a mode search and quadrature per gene.
///
/// ### References
///
/// Trefethen. *Approximation Theory and Approximation Practice.* SIAM (2013).
struct CountGroups {
    /// `(count, offsets)` of each interpolated group
    tabled: Vec<(u32, Vec<f64>)>,
    /// Counts of the swept genes
    y: Vec<u32>,
    /// Offsets of the swept genes
    off: Vec<f64>,
}

impl CountGroups {
    /// Split genes into interpolated groups and swept genes.
    ///
    /// ### Params
    ///
    /// * `y` - Counts
    /// * `off` - Per-gene offsets, same length
    /// * `n_evals` - Evaluations per gene
    /// * `spread` - Width added to each group's `t` range by the shifts it is
    ///   evaluated at (`ln(phi_max / phi_min)`, zero for one shift)
    ///
    /// ### Returns
    ///
    /// The split. A group is tabled when its `size * n_evals` evaluations
    /// exceed both [`MIN_TABLE_GENES`] and the interpolant's node count.
    fn new(y: &[u32], off: &[f64], n_evals: usize, spread: f64) -> Self {
        let n_tab = MAX_TABLED_COUNT as usize + 1;
        let mut size = vec![0usize; n_tab];
        let mut range = vec![(f64::INFINITY, f64::NEG_INFINITY); n_tab];
        for (&yg, &og) in y.iter().zip(off) {
            if (yg as usize) < n_tab {
                let k = yg as usize;
                size[k] += 1;
                range[k] = (range[k].0.min(og), range[k].1.max(og));
            }
        }
        let worth: Vec<bool> = (0..n_tab)
            .map(|k| {
                let evals = size[k] * n_evals;
                let width = range[k].1 - range[k].0 + spread;
                let nodes = ((width / CHEB_WIDTH).floor() as usize + 1) * (CHEB_DEGREE + 1);
                evals >= MIN_TABLE_GENES && evals > nodes
            })
            .collect();
        let tab = |yg: u32| (yg as usize) < n_tab && worth[yg as usize];
        let mut groups: Vec<Vec<f64>> = vec![Vec::new(); n_tab];
        let (mut ys, mut offs) = (Vec::new(), Vec::new());
        for (&yg, &og) in y.iter().zip(off) {
            if tab(yg) {
                groups[yg as usize].push(og);
            } else {
                ys.push(yg);
                offs.push(og);
            }
        }
        let tabled = groups
            .into_iter()
            .enumerate()
            .filter(|(_, g)| !g.is_empty())
            .map(|(x, g)| (x as u32, g))
            .collect();
        Self {
            tabled,
            y: ys,
            off: offs,
        }
    }
}

/////////////////
// FusedInterp //
/////////////////

/// Number of [`CHEB_WIDTH`] panels covering a `t` range.
///
/// ### Params
///
/// * `width` - `t_hi - t_lo`, `>= 0`
///
/// ### Returns
///
/// Panel count, `>= 1`.
#[inline]
fn n_panels(width: f64) -> usize {
    (width / CHEB_WIDTH).floor() as usize + 1
}

/// Chebyshev coefficients of [`ln_pln_fused`] on one panel.
///
/// ### Params
///
/// * `x` - Count
/// * `sigma` - Standard deviation of the latent log-rate
/// * `centre` - Panel centre in `t`
///
/// ### Returns
///
/// `[degree][form]`, so the panel's value at `u` in `[-1, 1]` is
/// `sum_k c_k T_k(u)`.
fn panel_coefs(x: u32, sigma: f64, centre: f64) -> [[f64; N_FORMS]; CHEB_DEGREE + 1] {
    let n = CHEB_DEGREE;
    let mut vals = [[0.0; N_FORMS]; CHEB_DEGREE + 1];
    for (j, v) in vals.iter_mut().enumerate() {
        let u = (PI * j as f64 / n as f64).cos();
        let d = ln_pln_fused(x, centre + 0.5 * CHEB_WIDTH * u, sigma);
        *v = [
            d.ln_p,
            d.d_mu,
            d.d_sigma,
            d.d2_mu,
            d.d2_mu_sigma,
            d.d2_sigma,
        ];
    }
    // c_k = (2 / n) sum'' f_j cos(pi j k / n), ends halved
    let mut c = [[0.0; N_FORMS]; CHEB_DEGREE + 1];
    for (k, ck) in c.iter_mut().enumerate() {
        for (j, v) in vals.iter().enumerate() {
            let w = if j == 0 || j == n { 0.5 } else { 1.0 };
            let cs = w * (PI * (j * k) as f64 / n as f64).cos();
            for f in 0..N_FORMS {
                ck[f] += cs * v[f];
            }
        }
        let w = if k == 0 || k == n { 1.0 } else { 2.0 };
        for f in 0..N_FORMS {
            ck[f] *= w / n as f64;
        }
    }
    c
}

/// Piecewise Chebyshev interpolant of [`ln_pln_fused`] in `t` at fixed count
/// and `sigma`.
struct FusedInterp {
    /// Left end of the first panel
    t_lo: f64,
    /// Coefficients, `[panel][degree][form]`
    coef: Vec<[[f64; N_FORMS]; CHEB_DEGREE + 1]>,
}

impl FusedInterp {
    /// Build over `[t_lo, t_hi]`.
    ///
    /// ### Params
    ///
    /// * `x` - Count
    /// * `sigma` - Standard deviation of the latent log-rate
    /// * `t_lo` - Smallest `t` to cover
    /// * `t_hi` - Largest `t` to cover
    ///
    /// ### Returns
    ///
    /// The interpolant.
    fn new(x: u32, sigma: f64, t_lo: f64, t_hi: f64) -> Self {
        let coef = (0..n_panels(t_hi - t_lo))
            .map(|p| panel_coefs(x, sigma, t_lo + (p as f64 + 0.5) * CHEB_WIDTH))
            .collect();
        Self { t_lo, coef }
    }

    /// Add the interpolated terms at `t` to an accumulator (Clenshaw).
    ///
    /// ### Params
    ///
    /// * `t` - Point inside the built range
    /// * `acc` - Accumulator, in [`PlnDerivs`] field order
    #[inline]
    fn add_to(&self, t: f64, acc: &mut [f64; N_FORMS]) {
        let r = (t - self.t_lo) / CHEB_WIDTH;
        let p = (r as usize).min(self.coef.len() - 1);
        let u = 2.0 * (r - p as f64) - 1.0;
        let c = &self.coef[p];
        let (mut b1, mut b2) = ([0.0; N_FORMS], [0.0; N_FORMS]);
        for ck in c[1..].iter().rev() {
            for f in 0..N_FORMS {
                let b0 = ck[f] + 2.0 * u * b1[f] - b2[f];
                b2[f] = b1[f];
                b1[f] = b0;
            }
        }
        for f in 0..N_FORMS {
            acc[f] += c[0][f] + u * b1[f] - b2[f];
        }
    }
}

/////////////////
// ChebMoments //
/////////////////

/// Chebyshev moments of one interpolated group in [`fit_lnpois`].
///
/// The fit builds its interpolant over `[mu + lo, mu + hi]` and evaluates it
/// at `mu + off_g`, so each gene's panel and local coordinate `u_g` do not
/// depend on `(mu, sigma)`. By linearity the group sum is
/// `sum_p sum_k c_pk M_pk` with `M_pk = sum_{g in p} T_k(u_g)`, computed once
/// per fit instead of a Clenshaw pass per gene and evaluation. Panels without
/// genes are never built.
struct ChebMoments {
    /// Count of the group
    x: u32,
    /// Smallest offset in the group
    lo: f64,
    /// `M_pk`, `[panel][degree]`
    m: Vec<[f64; CHEB_DEGREE + 1]>,
    /// Genes per panel
    size: Vec<usize>,
}

impl ChebMoments {
    /// Moments of one group.
    ///
    /// ### Params
    ///
    /// * `x` - Count
    /// * `offs` - Offsets of the group, non-empty
    ///
    /// ### Returns
    ///
    /// The moments.
    fn new(x: u32, offs: &[f64]) -> Self {
        let (lo, hi) = min_max(offs);
        let np = n_panels(hi - lo);
        let mut m = vec![[0.0; CHEB_DEGREE + 1]; np];
        let mut size = vec![0; np];
        for &o in offs {
            let r = (o - lo) / CHEB_WIDTH;
            let p = (r as usize).min(np - 1);
            let u = 2.0 * (r - p as f64) - 1.0;
            let mp = &mut m[p];
            let (mut t0, mut t1) = (1.0, u);
            mp[0] += t0;
            mp[1] += t1;
            for mk in mp[2..].iter_mut() {
                let t2 = 2.0 * u * t1 - t0;
                *mk += t2;
                (t0, t1) = (t1, t2);
            }
            size[p] += 1;
        }
        Self { x, lo, m, size }
    }

    /// Add the group's summed terms at `(mu, sigma)` to an accumulator.
    ///
    /// ### Params
    ///
    /// * `mu` - Mean of the latent log-rate
    /// * `sigma` - Standard deviation of the latent log-rate
    /// * `acc` - Accumulator, in [`PlnDerivs`] field order
    fn add_to(&self, mu: f64, sigma: f64, acc: &mut [f64; N_FORMS]) {
        for (p, mp) in self.m.iter().enumerate() {
            if self.size[p] == 0 {
                continue;
            }
            let c = panel_coefs(self.x, sigma, mu + self.lo + (p as f64 + 0.5) * CHEB_WIDTH);
            for (ck, &mk) in c.iter().zip(mp) {
                for f in 0..N_FORMS {
                    acc[f] += ck[f] * mk;
                }
            }
        }
    }
}

/// Smallest and largest value.
///
/// ### Params
///
/// * `v` - Values, non-empty
///
/// ### Returns
///
/// `(min, max)`.
fn min_max(v: &[f64]) -> (f64, f64) {
    v.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &o| {
            (a.min(o), b.max(o))
        })
}

/// [`sum_fused`] over the split genes: interpolated groups plus the sweep.
///
/// ### Params
///
/// * `g` - The split genes; only its sweep is read
/// * `tabs` - Moments of the interpolated groups of `g`
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
///
/// ### Returns
///
/// Summed loglikelihood, gradient and Hessian.
fn sum_grouped(g: &CountGroups, tabs: &[ChebMoments], mu: f64, sigma: f64) -> PlnDerivs {
    let mut acc = sum_fused(&g.y, &g.off, mu, sigma);
    let mut s = [0.0; N_FORMS];
    for t in tabs {
        t.add_to(mu, sigma, &mut s);
    }
    acc.add(&PlnDerivs {
        ln_p: s[0],
        d_mu: s[1],
        d_sigma: s[2],
        d2_mu: s[3],
        d2_mu_sigma: s[4],
        d2_sigma: s[5],
    });
    acc
}

/////////
// Fit //
/////////

/// Result of [`fit_lnpois`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LnpoisFit {
    /// Fitted mean of the latent log-rate
    pub mu: f64,
    /// Fitted standard deviation of the latent log-rate, `>= SIGMA_MIN`
    pub sigma: f64,
    /// Loglikelihood at the fit
    pub loglik: f64,
    /// Newton iterations used
    pub iterations: usize,
}

/// Method-of-moments start for [`fit_lnpois`].
///
/// With `m_g = exp(off_g)` and `k = E[e^z] = exp(mu + sigma^2 / 2)`:
/// `E[y_g] = k m_g` and `Var[y_g] = k m_g + k^2 m_g^2 (exp(sigma^2) - 1)`,
/// pooled over genes. Only moves Newton's start, not its optimum.
///
/// ### Params
///
/// * `y` - Counts
/// * `off` - Per-gene offsets `ln(d lambda_g)`
///
/// ### Returns
///
/// `(mu, sigma)`, with `sigma` clamped to `[SIGMA_MIN, MOMENT_SIGMA_MAX]`;
/// `(0, 1)` if the counts are all zero.
fn moment_start(y: &[u32], off: &[f64]) -> (f64, f64) {
    let (sy, sm) = y
        .iter()
        .zip(off)
        .fold((0.0, 0.0), |(a, b), (&yg, &o)| (a + yg as f64, b + o.exp()));
    if sy <= 0.0 {
        return (0.0, 1.0);
    }
    let k = sy / sm;
    let (num, den) = y.iter().zip(off).fold((0.0, 0.0), |(a, b), (&yg, &o)| {
        let m = k * o.exp();
        let r = yg as f64 - m;
        (a + r * r - m, b + m * m)
    });
    let s2 = (num / den).max(0.0).ln_1p();
    let sigma = s2.sqrt().clamp(SIGMA_MIN, MOMENT_SIGMA_MAX);
    (k.ln() - 0.5 * sigma * sigma, sigma)
}

/// Maximum-likelihood `(mu, sigma)` of the PLN at `phi = 1` (SPEC 4.3).
///
/// Bounded Newton from the moment start ([`moment_start`]) with `sigma >= SIGMA_MIN`, analytic gradient
/// and Hessian from [`ln_pln_fused`], and a backtracking line search. Where
/// the Hessian is not negative definite the step falls back to a diagonally
/// scaled gradient step. On the bound with the gradient pointing out,
/// `sigma` is held and only `mu` moves.
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, same length as `y`, all `> 0`
/// * `d` - Library size
///
/// ### Returns
///
/// [`LnpoisFit`], or an input or convergence error.
pub fn fit_lnpois<T: CnvFloat>(y: &[u32], lambda: &[T], d: f64) -> Result<LnpoisFit, CnvErrors> {
    if y.is_empty() {
        return Err(CnvErrors::EmptyInput {
            what: "genes for fit_lnpois",
        });
    }
    let off = offsets(y, lambda, d, None)?;
    let (mut mu, mut sigma) = moment_start(y, &off);
    let groups = CountGroups::new(y, &off, 1, 0.0);
    let tabs: Vec<ChebMoments> = groups
        .tabled
        .iter()
        .map(|(x, offs)| ChebMoments::new(*x, offs))
        .collect();
    let mut cur = sum_grouped(&groups, &tabs, mu, sigma);
    for iter in 0..FIT_MAX_ITER {
        let (g1, g2) = (cur.d_mu, cur.d_sigma);
        let (h11, h12, h22) = (cur.d2_mu, cur.d2_mu_sigma, cur.d2_sigma);
        let at_bound = sigma <= SIGMA_MIN && g2 <= 0.0;
        let det = h11 * h22 - h12 * h12;
        let (d1, d2) = if at_bound {
            (
                if h11 < 0.0 {
                    -g1 / h11
                } else {
                    g1 / h11.abs().max(1.0)
                },
                0.0,
            )
        } else if h11 < 0.0 && det > 0.0 {
            ((-h22 * g1 + h12 * g2) / det, (h12 * g1 - h11 * g2) / det)
        } else {
            (g1 / h11.abs().max(1.0), g2 / h22.abs().max(1.0))
        };
        let pred = g1 * d1 + g2 * d2;
        if pred <= FIT_TOL * (1.0 + cur.ln_p.abs()) {
            return Ok(LnpoisFit {
                mu,
                sigma,
                loglik: cur.ln_p,
                iterations: iter,
            });
        }
        let mut t = 1.0;
        let mut accepted = false;
        for _ in 0..FIT_MAX_HALVINGS {
            let (mu_n, sig_n) = (mu + t * d1, (sigma + t * d2).max(SIGMA_MIN));
            let next = sum_grouped(&groups, &tabs, mu_n, sig_n);
            if next.ln_p > cur.ln_p {
                (mu, sigma, cur) = (mu_n, sig_n, next);
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            // no ascent left at machine precision
            return Ok(LnpoisFit {
                mu,
                sigma,
                loglik: cur.ln_p,
                iterations: iter,
            });
        }
    }
    Err(CnvErrors::NonConvergence {
        what: "fit_lnpois",
        iterations: FIT_MAX_ITER,
    })
}

/////////
// Phi //
/////////

/// Maximise `l(phi)` over `[lo, hi]` from `phi = 1`, in `t = ln phi`.
///
/// `ln PLN` is log-concave in `mu` (Prekopa), so `l` is concave in `t`.
///
/// ### Params
///
/// * `y` - Counts
/// * `off` - Per-gene offsets `ln(d lambda_g)`
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
/// * `bounds` - `(lo, hi)` on `phi`
/// * `what` - Name for errors
///
/// ### Returns
///
/// `phi_hat`, or a convergence error.
fn phi_argmax(
    y: &[u32],
    off: &[f64],
    mu: f64,
    sigma: f64,
    bounds: (f64, f64),
    what: &'static str,
) -> Result<f64, CnvErrors> {
    let t = maximise_concave_1d(
        |t| {
            let s = sum_fused(y, off, mu + t, sigma);
            (s.d_mu, s.d2_mu)
        },
        bounds.0.ln(),
        bounds.1.ln(),
        0.0,
        PHI_TOL,
        PHI_MAX_ITER,
        what,
    )?;
    // exact bounds, since exp(ln b) != b in general
    Ok(if t <= bounds.0.ln() {
        bounds.0
    } else if t >= bounds.1.ln() {
        bounds.1
    } else {
        t.exp()
    })
}

/// MLE of the fold change `phi` on [`PHI_MLE_BOUNDS`] (SPEC 4.4).
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, same length as `y`, all `> 0`
/// * `d` - Library size
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
///
/// ### Returns
///
/// `phi_hat`; `1` for an empty gene set (N/R/utils.R:1427).
pub fn phi_mle<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    mu: f64,
    sigma: f64,
) -> Result<f64, CnvErrors> {
    let off = offsets(y, lambda, d, Some(sigma))?;
    if y.is_empty() {
        return Ok(1.0);
    }
    phi_argmax(y, &off, mu, sigma, PHI_MLE_BOUNDS, "phi_mle")
}

/// Laplace approximation to the posterior of `phi` (SPEC 4.5).
///
/// `phi_hat = argmax l(phi)` on [`PHI_POST_BOUNDS`] and
/// `s_phi = (-d^2 l / d phi^2)^{-1/2}` at `phi_hat`, from the analytic
/// derivatives: `d^2 l / d phi^2 = (l_tt - l_t) / phi^2` with `t = ln phi`.
///
/// ### Params
///
/// * `y` - Counts
/// * `lambda` - Reference proportions, same length as `y`, all `> 0`
/// * `d` - Library size
/// * `mu` - Mean of the latent log-rate
/// * `sigma` - Standard deviation of the latent log-rate
///
/// ### Returns
///
/// `(phi_hat, s_phi)`; `(1, 0)` for an empty gene set, a failed fit or a
/// non-finite `s_phi` (N/R/utils.R:1510, 1530-1535).
pub fn approx_phi_post<T: CnvFloat>(
    y: &[u32],
    lambda: &[T],
    d: f64,
    mu: f64,
    sigma: f64,
) -> Result<(f64, f64), CnvErrors> {
    let off = offsets(y, lambda, d, Some(sigma))?;
    if y.is_empty() {
        return Ok((1.0, 0.0));
    }
    let Ok(phi) = phi_argmax(y, &off, mu, sigma, PHI_POST_BOUNDS, "approx_phi_post") else {
        return Ok((1.0, 0.0));
    };
    let s = sum_fused(y, &off, mu + phi.ln(), sigma);
    let h = -(s.d2_mu - s.d_mu) / (phi * phi);
    let sd = (1.0 / h).sqrt();
    if sd.is_finite() {
        Ok((phi, sd))
    } else {
        Ok((1.0, 0.0))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/fixtures");

    /// Rows of a TSV fixture as strings, header dropped.
    fn read_tsv(name: &str) -> Vec<Vec<String>> {
        let s = std::fs::read_to_string(format!("{FIXTURES}/{name}"))
            .unwrap_or_else(|e| panic!("{name}: {e}; run reference/pln.R"));
        s.lines()
            .skip(1)
            .map(|l| l.split('\t').map(|c| c.trim().to_string()).collect())
            .collect()
    }

    /// Parse an R number, including `-Inf` and `NA`.
    fn num(s: &str) -> f64 {
        match s {
            "-Inf" => f64::NEG_INFINITY,
            "Inf" => f64::INFINITY,
            "NA" => f64::NAN,
            _ => s.parse().unwrap(),
        }
    }

    /// One simulated gene set.
    struct GeneSet {
        /// Counts
        y: Vec<u32>,
        /// Reference proportions
        lambda: Vec<f64>,
        /// Library size
        d: f64,
        /// Every value in `pln_sets_meta.tsv` for this set, by column name
        meta: std::collections::HashMap<String, f64>,
    }

    /// All 12 gene sets from `pln_genes_n*.tsv` and `pln_sets_meta.tsv`.
    fn gene_sets() -> Vec<(String, GeneSet)> {
        let meta_raw = std::fs::read_to_string(format!("{FIXTURES}/pln_sets_meta.tsv")).unwrap();
        let mut lines = meta_raw.lines();
        let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let mut out = Vec::new();
        for l in lines {
            let cells: Vec<&str> = l.split('\t').map(str::trim).collect();
            let set = cells[0].to_string();
            let meta: std::collections::HashMap<String, f64> = header
                .iter()
                .zip(&cells)
                .skip(1)
                .map(|(h, c)| (h.to_string(), num(c)))
                .collect();
            let n = meta["n"] as usize;
            let col = match set.split_once('_').unwrap().1 {
                "base" => 1,
                "phi0.5" => 2,
                "phi1" => 3,
                "phi1.5" => 4,
                other => panic!("unknown set {other}"),
            };
            let rows = read_tsv(&format!("pln_genes_n{n}.tsv"));
            let y = rows.iter().map(|r| r[col].parse().unwrap()).collect();
            let lambda = rows.iter().map(|r| num(&r[0])).collect();
            let d = meta["d"];
            out.push((set, GeneSet { y, lambda, d, meta }));
        }
        out
    }

    #[test]
    fn test_gauss_hermite_moments() {
        let (t, w) = gauss_hermite(GH_NODES);
        let sp = PI.sqrt();
        assert_relative_eq!(w.iter().sum::<f64>(), sp, max_relative = 1e-13);
        let m2: f64 = t.iter().zip(&w).map(|(t, w)| w * t * t).sum();
        assert_relative_eq!(m2, sp / 2.0, max_relative = 1e-13);
    }

    #[test]
    fn test_ln_factorial_table_and_stirling() {
        let mut exact = 0.0;
        for k in 1..=2000u32 {
            exact += (k as f64).ln();
            assert_relative_eq!(ln_factorial(k), exact, max_relative = 1e-13);
        }
    }

    #[test]
    fn test_ln_pln_vs_tight_reference() {
        let mut max_rel = 0.0f64;
        let rows = [read_tsv("pln_grid.tsv"), read_tsv("pln_random.tsv")].concat();
        for r in rows {
            let (x, mu, sigma) = (r[0].parse().unwrap(), num(&r[1]), num(&r[2]));
            let reference = num(&r[3]);
            let mut terms = [0.0; GH_NODES];
            let raw = pln_quadrature(x, mu, sigma, &mut terms).0;
            // relative error in p
            max_rel = max_rel.max((raw - reference).exp_m1().abs());
            assert_eq!(ln_pln(x, mu, sigma), raw);
        }
        println!("max relative error vs tight reference: {max_rel:.3e}");
        assert!(max_rel <= 1e-6, "max relative error {max_rel:e}");
    }

    #[test]
    fn test_ln_pln_vs_poilog_and_hahmmr() {
        // Both oracles use QAGS at 1e-5 and are off the tight reference by up
        // to 4e-4 (hahmmr) and 0.27 (poilog). Ours must sit no further from
        // either than the tight reference does, plus 1e-6.
        let (mut max_poilog, mut max_hahmmr) = (0.0f64, 0.0f64);
        for r in read_tsv("pln_grid.tsv") {
            let (x, mu, sigma) = (r[0].parse().unwrap(), num(&r[1]), num(&r[2]));
            let reference = num(&r[3]);
            // hahmmr floors underflowed p at 1e-15 and poilog returns 0 there
            if reference < 1e-15f64.ln() {
                continue;
            }
            let ours = ln_pln(x, mu, sigma);
            for (col, max) in [(5, &mut max_poilog), (6, &mut max_hahmmr)] {
                let oracle = num(&r[col]);
                let d_ours = (ours - oracle).exp_m1().abs();
                *max = max.max(d_ours);
                assert!(d_ours <= (reference - oracle).exp_m1().abs() + 1e-6);
            }
        }
        println!("max relative difference: poilog {max_poilog:.3e}, hahmmr {max_hahmmr:.3e}");
    }

    #[test]
    fn test_ln_pln_fused_matches_finite_differences() {
        let h = 1e-4;
        for &(x, mu, sigma) in &[
            (0u32, -3.0, 0.5),
            (0, 2.0, 2.0),
            (1, -1.0, 0.8),
            (5, 0.0, 1.0),
            (20, 2.0, 0.1),
            (100, 6.0, 0.5),
        ] {
            let f = ln_pln_fused(x, mu, sigma);
            assert_eq!(f.ln_p, ln_pln(x, mu, sigma));
            let l = |m: f64, s: f64| {
                let mut terms = [0.0; GH_NODES];
                pln_quadrature(x, m, s, &mut terms).0
            };
            let g_mu = (l(mu + h, sigma) - l(mu - h, sigma)) / (2.0 * h);
            let g_s = (l(mu, sigma + h) - l(mu, sigma - h)) / (2.0 * h);
            // central difference of the analytic gradient along (dm, ds)
            let fd = |dm: f64, ds: f64| {
                let a = ln_pln_fused(x, mu + dm * h, sigma + ds * h);
                let b = ln_pln_fused(x, mu - dm * h, sigma - ds * h);
                (
                    (a.d_mu - b.d_mu) / (2.0 * h),
                    (a.d_sigma - b.d_sigma) / (2.0 * h),
                )
            };
            let (h11, h21) = fd(1.0, 0.0);
            let (h12, h22) = fd(0.0, 1.0);
            let tol = 1e-5;
            assert_relative_eq!(f.d_mu, g_mu, epsilon = tol, max_relative = tol);
            assert_relative_eq!(f.d_sigma, g_s, epsilon = tol, max_relative = tol);
            assert_relative_eq!(f.d2_mu, h11, epsilon = tol, max_relative = tol);
            assert_relative_eq!(f.d2_mu_sigma, h21, epsilon = tol, max_relative = tol);
            assert_relative_eq!(f.d2_mu_sigma, h12, epsilon = tol, max_relative = tol);
            assert_relative_eq!(f.d2_sigma, h22, epsilon = tol, max_relative = tol);
        }
    }

    #[test]
    fn test_ln_pln_no_floor_far_tail() {
        let f = ln_pln_fused(1000, -8.0, 0.01);
        assert!(f.ln_p < -1e4 && f.ln_p.is_finite());
        assert!(f.d_mu.is_finite() && f.d2_sigma.is_finite());
    }

    #[test]
    fn test_l_lnpois_vs_reference_and_hahmmr() {
        let (mut max_ref, mut max_hahmmr) = (0.0f64, 0.0f64);
        for (_, s) in gene_sets() {
            let ours = l_lnpois(&s.y, &s.lambda, s.d, -0.5, 0.7, 1.3).unwrap();
            let r = s.meta["l_fixed_ref"];
            max_ref = max_ref.max(((ours - r) / r).abs());
            max_hahmmr = max_hahmmr.max(((ours - s.meta["l_fixed"]) / r).abs());
        }
        println!("l_lnpois max relative difference: tight {max_ref:.3e}, hahmmr {max_hahmmr:.3e}");
        assert!(max_ref < 1e-9);
        assert!(max_hahmmr < 1e-5);
    }

    #[test]
    fn test_l_lnpois_multi_matches_single() {
        let (_, s) = gene_sets().swap_remove(0);
        let phis = [1.0, 0.5, 1.5, 2.0, 0.25];
        let multi = l_lnpois_multi(&s.y, &s.lambda, s.d, -1.0, 0.8, &phis).unwrap();
        for (k, &phi) in phis.iter().enumerate() {
            let single = l_lnpois(&s.y, &s.lambda, s.d, -1.0, 0.8, phi).unwrap();
            assert_relative_eq!(multi[k], single, max_relative = 1e-12);
        }
        let f32_lambda: Vec<f32> = s.lambda.iter().map(|&l| l as f32).collect();
        let l32 = l_lnpois(&s.y, &f32_lambda, s.d, -1.0, 0.8, 1.0).unwrap();
        assert_relative_eq!(l32, multi[0], max_relative = 1e-6);
    }

    #[test]
    fn test_l_lnpois_rejects_bad_input() {
        let y = [0u32, 1];
        assert!(l_lnpois(&y, &[0.5f64], 10.0, 0.0, 1.0, 1.0).is_err());
        assert!(l_lnpois(&y, &[0.5f64, 0.0], 10.0, 0.0, 1.0, 1.0).is_err());
        assert!(l_lnpois(&y, &[0.5f64, 0.5], 10.0, 0.0, 0.0, 1.0).is_err());
        assert!(l_lnpois(&y, &[0.5f64, 0.5], 10.0, 0.0, 1.0, -1.0).is_err());
        assert!(l_lnpois(&y, &[0.5f64, 0.5], 0.0, 0.0, 1.0, 1.0).is_err());
        assert_eq!(l_lnpois::<f64>(&[], &[], 10.0, 0.0, 1.0, 1.0).unwrap(), 0.0);
    }

    #[test]
    fn test_fit_lnpois_vs_hahmmr() {
        for (name, s) in gene_sets() {
            if !name.ends_with("base") {
                continue;
            }
            let fit = fit_lnpois(&s.y, &s.lambda, s.d).unwrap();
            let (r_mu, r_sig, r_l) = (s.meta["fit_mu"], s.meta["fit_sigma"], s.meta["fit_loglik"]);
            let ours_at_r = l_lnpois(&s.y, &s.lambda, s.d, r_mu, r_sig, 1.0).unwrap();
            println!(
                "{name}: iter {}, l {:.10} vs R {:.10} (ours at R's par {:.10}), \
                 dmu {:.3e}, dsigma {:.3e}",
                fit.iterations,
                fit.loglik,
                r_l,
                ours_at_r,
                fit.mu - r_mu,
                fit.sigma - r_sig
            );
            assert!(fit.loglik >= r_l - 1e-6 * r_l.abs());
            assert!(fit.loglik >= ours_at_r);
            assert!(fit.sigma >= SIGMA_MIN);
        }
    }

    #[test]
    fn test_fit_lnpois_edge_sets_vs_hahmmr() {
        use crate::reference::tests::read_tsv;
        let lambda: Vec<f64> = read_tsv("pln_fit_edge_lambda.tsv")
            .iter()
            .map(|r| r[0].trim().parse().unwrap())
            .collect();
        for r in read_tsv("pln_fit_edge.tsv") {
            let f = |i: usize| -> f64 { r[i].trim().parse().unwrap() };
            let d = f(1);
            let (r_mu, r_sig, r_l) = (f(2), f(3), f(4));
            let y: Vec<u32> = r[5].split(',').map(|v| v.parse().unwrap()).collect();
            let fit = fit_lnpois(&y, &lambda, d).unwrap();
            let ours_at_r = l_lnpois(&y, &lambda, d, r_mu, r_sig, 1.0).unwrap();
            println!(
                "{}: iter {}, mu {:.6} sigma {:.6} l {:.10} | R mu {:.6} sigma {:.6} l {:.10}",
                r[0], fit.iterations, fit.mu, fit.sigma, fit.loglik, r_mu, r_sig, r_l
            );
            // never worse than hahmmr's optimum, scored by either quadrature
            assert!(fit.loglik >= r_l - 1e-6 * r_l.abs());
            assert!(fit.loglik >= ours_at_r - 1e-9 * ours_at_r.abs());
        }
    }

    #[test]
    fn test_fit_lnpois_sigma_bound() {
        // constant counts at the expected rate push sigma to its bound
        let n = 200;
        let lambda = vec![1.0 / n as f64; n];
        let y = vec![5u32; n];
        let fit = fit_lnpois(&y, &lambda, 5.0 * n as f64).unwrap();
        assert_eq!(fit.sigma, SIGMA_MIN);
        let l_up = l_lnpois(&y, &lambda, 5.0 * n as f64, fit.mu, 0.02, 1.0).unwrap();
        assert!(fit.loglik >= l_up);
    }

    #[test]
    fn test_phi_vs_numbat() {
        for (name, s) in gene_sets() {
            if name.ends_with("base") {
                continue;
            }
            let (mu, sigma) = (s.meta["fit_mu"], s.meta["fit_sigma"]);
            let mle = phi_mle(&s.y, &s.lambda, s.d, mu, sigma).unwrap();
            let (post, sd) = approx_phi_post(&s.y, &s.lambda, s.d, mu, sigma).unwrap();
            let rel = |a: f64, b: f64| ((a - b) / b).abs();
            let (r_mle, r_post, r_sd) = (
                s.meta["phi_mle"],
                s.meta["phi_post_mean"],
                s.meta["phi_post_sd"],
            );
            println!(
                "{name}: phi_mle {mle:.8} ({:.2e}), phi_hat {post:.8} ({:.2e}), \
                 s_phi {sd:.8} ({:.2e})",
                rel(mle, r_mle),
                rel(post, r_post),
                rel(sd, r_sd)
            );
            assert!(rel(mle, r_mle) < 1e-4);
            assert!(rel(post, r_post) < 1e-4);
            assert!(rel(sd, r_sd) < 1e-4);
        }
    }

    #[test]
    fn test_phi_empty_and_boundary() {
        assert_eq!(phi_mle::<f64>(&[], &[], 10.0, 0.0, 1.0).unwrap(), 1.0);
        assert_eq!(
            approx_phi_post::<f64>(&[], &[], 10.0, 0.0, 1.0).unwrap(),
            (1.0, 0.0)
        );
        // all zeros: l(phi) decreases in phi, so phi_hat sits on the lower bound
        let y = vec![0u32; 50];
        let lambda = vec![0.02f64; 50];
        assert_eq!(
            phi_mle(&y, &lambda, 1000.0, 0.0, 0.5).unwrap(),
            PHI_MLE_BOUNDS.0
        );
    }

    #[test]
    fn test_fused_interp_matches_quadrature() {
        let mut worst = [0.0f64; N_FORMS];
        for &x in &[0u32, 1, 2, 3] {
            for &sigma in &[0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0] {
                let (lo, hi) = (-12.0, 4.0);
                let it = FusedInterp::new(x, sigma, lo, hi);
                for k in 0..=1601 {
                    let t = lo + (hi - lo) * k as f64 / 1601.0;
                    let mut a = [0.0; N_FORMS];
                    it.add_to(t, &mut a);
                    let d = ln_pln_fused(x, t, sigma);
                    let e = [
                        d.ln_p,
                        d.d_mu,
                        d.d_sigma,
                        d.d2_mu,
                        d.d2_mu_sigma,
                        d.d2_sigma,
                    ];
                    for f in 0..N_FORMS {
                        let err = (a[f] - e[f]).abs() / e[f].abs().max(1.0);
                        worst[f] = worst[f].max(err);
                    }
                }
            }
        }
        println!("worst {:?}", worst.map(|w| format!("{w:.2e}")));
        // ln p, then the derivatives
        let tol = [1e-9, 1e-9, 1e-8, 1e-8, 1e-8, 1e-6];
        for f in 0..N_FORMS {
            assert!(worst[f] < tol[f], "form {f}: {:.2e}", worst[f]);
        }
    }
}
