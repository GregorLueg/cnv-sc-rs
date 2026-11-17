//! SIMD kernels. The only module that names `wide` types.
//!
//! `f64x4` lowers to two NEON `f64x2` on aarch64 and to SSE2 pairs on x86-64
//! without target features. Lane sums are reduced in a fixed order, so results
//! do not depend on the thread count.

use ::wide::bytemuck::cast;
use ::wide::{f64x4, i64x4};

/// Lanes per vector.
const LANES: usize = 4;

/// Lower clamp on exponent arguments: keeps `2^r` normal, so no subnormal
/// path. `exp(-708)` is about 3e-308; our choice.
const EXP_ARG_MIN: f64x4 = f64x4::splat(-708.0);

/// High part of `ln 2` for the two-step reduction (fdlibm `e_exp.c`).
const LN2_HI: f64x4 = f64x4::splat(0.693_145_751_953_125);

/// Low part of `ln 2`, `LN2_HI + LN2_LO = ln 2` to full precision.
const LN2_LO: f64x4 = f64x4::splat(1.428_606_820_309_417_2e-6);

/// `log2(e)`.
const LOG2_E: f64x4 = f64x4::splat(std::f64::consts::LOG2_E);

/// `1.5 * 2^52`: adding it rounds to an integer held in the low mantissa
/// bits. `wide`'s `round` has no NEON path.
const SHIFT: f64x4 = f64x4::splat(6_755_399_441_055_744.0);

/// IEEE double exponent bias.
const EXP_BIAS: i64x4 = i64x4::splat(1023);

/// `1 / k!` for `k = 2..=13`, the reduced-argument Taylor series. Degree 13
/// on `|y| <= ln(2) / 2` truncates at about 4e-18 relative, as in `wide`.
const INV_FACT: [f64x4; 12] = [
    f64x4::splat(1.0 / 2.0),
    f64x4::splat(1.0 / 6.0),
    f64x4::splat(1.0 / 24.0),
    f64x4::splat(1.0 / 120.0),
    f64x4::splat(1.0 / 720.0),
    f64x4::splat(1.0 / 5040.0),
    f64x4::splat(1.0 / 40320.0),
    f64x4::splat(1.0 / 362880.0),
    f64x4::splat(1.0 / 3628800.0),
    f64x4::splat(1.0 / 39916800.0),
    f64x4::splat(1.0 / 479001600.0),
    f64x4::splat(1.0 / 6227020800.0),
];

/// Range-reduced exponential without special cases: `x = r ln 2 + y`.
///
/// Inputs must be finite and below about 709; arguments under
/// [`EXP_ARG_MIN`] are clamped to it. No NaN, overflow or subnormal handling,
/// which is what makes it cheaper than `wide`'s `exp`. Constants are `const`
/// items: `f64x4::splat` inside the loop compiled to a `memset_pattern16`
/// call per constant and iteration on aarch64.
///
/// ### Params
///
/// * `x` - Argument
///
/// ### Returns
///
/// `(expm1(y), 2^r)`, so `exp(x) = 2^r (expm1(y) + 1)` and
/// `expm1(x) = 2^r expm1(y) + (2^r - 1)`.
#[inline(always)]
fn exp_parts(x: f64x4) -> (f64x4, f64x4) {
    let x = x.max(EXP_ARG_MIN);
    let k = x.mul_add(LOG2_E, SHIFT);
    let r = k - SHIFT;
    let y = r.mul_neg_add(LN2_HI, x);
    let y = r.mul_neg_add(LN2_LO, y);
    let mut q = INV_FACT[11];
    for &c in INV_FACT[..11].iter().rev() {
        q = q.mul_add(y, c);
    }
    let p = (y * y).mul_add(q, y);
    // 2^r from the low bits of k; r >= -1021 after the clamp
    let two_r: f64x4 = cast((cast::<_, i64x4>(k) + EXP_BIAS) << 52);
    (p, two_r)
}

/// Unnormalised Gauss-Hermite node masses of the PLN log-integrand.
///
/// For each node `v_i = s z_i`:
/// `dh_i = x v_i - e^m expm1(v_i) - v_i (v_i + 2c) / (2 sigma^2)` and
/// `terms_i = w_i exp(dh_i)`, as in `pln_quadrature` (SPEC 4.1).
///
/// ### Params
///
/// * `xf` - Count as `f64`
/// * `em` - `e^m` at the mode `m`
/// * `s` - Node scale at the mode
/// * `c` - `m - mu`
/// * `half_iv` - `1 / (2 sigma^2)`
/// * `z` - Scaled Hermite nodes
/// * `w` - Scaled Hermite weights, same length
/// * `terms` - Receives the node masses, same length
///
/// ### Returns
///
/// Sum of `terms`, lanes reduced in a fixed order.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn pln_node_masses(
    xf: f64,
    em: f64,
    s: f64,
    c: f64,
    half_iv: f64,
    z: &[f64],
    w: &[f64],
    terms: &mut [f64],
) -> f64 {
    let (vx, vem, vs, vhiv, vc2) = (
        f64x4::splat(xf),
        f64x4::splat(em),
        f64x4::splat(s),
        f64x4::splat(half_iv),
        f64x4::splat(2.0 * c),
    );
    let mut acc = f64x4::ZERO;
    let n = z.len() / LANES * LANES;
    for ((zc, wc), tc) in z[..n]
        .chunks_exact(LANES)
        .zip(w[..n].chunks_exact(LANES))
        .zip(terms[..n].chunks_exact_mut(LANES))
    {
        let v = vs * f64x4::from(<[f64; LANES]>::try_from(zc).expect("chunk of LANES"));
        let (p, two_r) = exp_parts(v);
        let em1 = two_r.mul_add(p, two_r - f64x4::ONE);
        let dh = vx * v - vem * em1 - v * (v + vc2) * vhiv;
        let (p, two_r) = exp_parts(dh);
        let ex = two_r.mul_add(p, two_r);
        let t = f64x4::from(<[f64; LANES]>::try_from(wc).expect("chunk of LANES")) * ex;
        tc.copy_from_slice(&t.to_array());
        acc += t;
    }
    let mut sum = acc.reduce_add();
    for i in n..z.len() {
        let v = s * z[i];
        let dh = xf * v - em * v.exp_m1() - v * (v + 2.0 * c) * half_iv;
        terms[i] = w[i] * dh.exp();
        sum += terms[i];
    }
    sum
}
