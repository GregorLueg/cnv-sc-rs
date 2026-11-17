//! SIMD kernels. The only module that names `wide` types or `std::arch`
//! intrinsics.
//!
//! x86-64 picks a kernel at runtime ([`detect_simd_level`]): AVX-512, AVX2 with
//! FMA, or `wide`. The 256- and 512-bit kernels are hand-written against
//! `std::arch`, because `wide` picks its width from a compile-time
//! `cfg(target_feature)`: on the x86-64 baseline its `f64x4` is two SSE2
//! halves, without FMA. aarch64 always takes the `wide` path (NEON).
//!
//! Lane sums are reduced in a fixed order, so results do not depend on the
//! thread count. They can differ in the last bits between levels (lane count,
//! and FMA versus separate multiply and add).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

use ::wide::bytemuck::cast;
use ::wide::{f64x4, i64x4};

////////////
// Consts //
////////////

/// Lower clamp on exponent arguments: keeps `2^r` normal, so no subnormal
/// path. `exp(-708)` is about 3e-308; our choice.
const EXP_ARG_MIN: f64 = -708.0;

/// High part of `ln 2` for the two-step reduction (fdlibm `e_exp.c`).
const LN2_HI: f64 = 0.693_145_751_953_125;

/// Low part of `ln 2`, `LN2_HI + LN2_LO = ln 2` to full precision.
const LN2_LO: f64 = 1.428_606_820_309_417_2e-6;

/// `1.5 * 2^52`: adding it rounds to an integer held in the low mantissa
/// bits. `wide`'s `round` has no NEON path.
const SHIFT: f64 = 6_755_399_441_055_744.0;

/// IEEE double exponent bias.
const EXP_BIAS: i64 = 1023;

/// `1 / k!` for `k = 2..=13`, the reduced-argument Taylor series. Degree 13
/// on `|y| <= ln(2) / 2` truncates at about 4e-18 relative, as in `wide`.
const INV_FACT: [f64; 12] = [
    1.0 / 2.0,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
];

///////////////
// SimdLevel //
///////////////

/// Kernel width, ordered so a request can be clamped to what the CPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdLevel {
    /// 128-bit `wide` (SSE2 on x86-64, NEON on aarch64)
    Sse,
    /// 256-bit AVX2 with FMA
    Avx2,
    /// 512-bit AVX-512F
    Avx512,
}

/// Detected level, probed once.
#[cfg(target_arch = "x86_64")]
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Widest SIMD level this CPU runs, lowered by `CNV_SC_SIMD` if set.
///
/// A shipped crate cannot assume AVX2 without an illegal-instruction crash on
/// older hardware, hence the runtime probe. FMA is probed with AVX2 because
/// the AVX2 kernel uses it.
///
/// ### Returns
///
/// The level the kernels dispatch on.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        let detected = if is_x86_feature_detected!("avx512f") {
            SimdLevel::Avx512
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            SimdLevel::Avx2
        } else {
            SimdLevel::Sse
        };
        cap_simd_level(detected)
    })
}

/// Widest SIMD level this CPU runs. NEON is the aarch64 baseline, so the
/// dispatch folds away at compile time.
///
/// ### Returns
///
/// [`SimdLevel::Sse`].
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn detect_simd_level() -> SimdLevel {
    SimdLevel::Sse
}

/// Lower the detected level if `CNV_SC_SIMD` (`sse`, `avx2`, `avx512`) asks
/// for it.
///
/// Downgrade only: naming a level the CPU lacks would run an illegal
/// instruction. Exists so one path can be timed or diffed against another on
/// the same machine.
///
/// ### Params
///
/// * `detected` - The widest level this CPU supports
///
/// ### Returns
///
/// The requested level when it is no wider than `detected`, else `detected`.
#[cfg(target_arch = "x86_64")]
fn cap_simd_level(detected: SimdLevel) -> SimdLevel {
    let Ok(request) = std::env::var("CNV_SC_SIMD") else {
        return detected;
    };
    let requested = match request.trim().to_lowercase().as_str() {
        "" => return detected,
        "sse" => SimdLevel::Sse,
        "avx2" => SimdLevel::Avx2,
        "avx512" => SimdLevel::Avx512,
        other => {
            eprintln!("CNV_SC_SIMD: unknown level {other:?}, using {detected:?}");
            return detected;
        }
    };
    if requested > detected {
        eprintln!("CNV_SC_SIMD: this CPU cannot run {requested:?}, using {detected:?}");
        return detected;
    }
    requested
}

////////////////
// Node masses //
////////////////

/// Unnormalised Gauss-Hermite node masses of the PLN log-integrand.
///
/// For each node `v_i = s z_i`:
/// `dh_i = x v_i - e^m expm1(v_i) - v_i (v_i + 2c) / (2 sigma^2)` and
/// `terms_i = w_i exp(dh_i)`, as in `pln_quadrature` (SPEC 4.1). `exp` and
/// `expm1` are range-reduced Taylor series without special cases (see
/// [`exp_parts_128`]); within about 1 ulp of libm.
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
    let p = NodeParams {
        xf,
        em,
        s,
        c,
        half_iv,
    };
    match detect_simd_level() {
        SimdLevel::Sse => node_masses_128(&p, z, w, terms),
        #[cfg(target_arch = "x86_64")]
        // SAFETY: detect_simd_level returns Avx2 only if avx2 and fma are present
        SimdLevel::Avx2 => unsafe { node_masses_avx2(&p, z, w, terms) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: detect_simd_level returns Avx512 only if avx512f is present
        SimdLevel::Avx512 => unsafe { node_masses_avx512(&p, z, w, terms) },
        #[cfg(not(target_arch = "x86_64"))]
        _ => unreachable!("only Sse is detected off x86-64"),
    }
}

/// Per-call scalars of [`pln_node_masses`].
struct NodeParams {
    /// Count as `f64`
    xf: f64,
    /// `e^m` at the mode
    em: f64,
    /// Node scale at the mode
    s: f64,
    /// `m - mu`
    c: f64,
    /// `1 / (2 sigma^2)`
    half_iv: f64,
}

/// Scalar node masses for the tail that does not fill a vector.
///
/// ### Params
///
/// * `p` - Per-call scalars
/// * `z` - Scaled Hermite nodes of the tail
/// * `w` - Weights of the tail
/// * `terms` - Receives the tail's node masses
///
/// ### Returns
///
/// Sum of the tail's `terms`.
#[inline(always)]
fn node_masses_tail(p: &NodeParams, z: &[f64], w: &[f64], terms: &mut [f64]) -> f64 {
    let mut sum = 0.0;
    for ((&zi, &wi), t) in z.iter().zip(w).zip(terms) {
        let v = p.s * zi;
        let dh = p.xf * v - p.em * v.exp_m1() - v * (v + 2.0 * p.c) * p.half_iv;
        *t = wi * dh.exp();
        sum += *t;
    }
    sum
}

//////////////////
// 128-bit wide //
//////////////////

/// Range-reduced exponential without special cases: `x = r ln 2 + y`.
///
/// Inputs must be finite and below about 709; arguments under
/// [`EXP_ARG_MIN`] are clamped to it. No NaN, overflow or subnormal handling,
/// which is what makes it cheaper than `wide`'s `exp`. Vector constants are
/// `const` items: `f64x4::splat` inside the loop compiled to a
/// `memset_pattern16` call per constant and iteration on aarch64.
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
fn exp_parts_128(x: f64x4) -> (f64x4, f64x4) {
    const MIN: f64x4 = f64x4::splat(EXP_ARG_MIN);
    const HI: f64x4 = f64x4::splat(LN2_HI);
    const LO: f64x4 = f64x4::splat(LN2_LO);
    const L2E: f64x4 = f64x4::splat(std::f64::consts::LOG2_E);
    const SH: f64x4 = f64x4::splat(SHIFT);
    const BIAS: i64x4 = i64x4::splat(EXP_BIAS);
    const IF: [f64x4; 12] = {
        let mut a = [f64x4::ZERO; 12];
        let mut i = 0;
        while i < 12 {
            a[i] = f64x4::splat(INV_FACT[i]);
            i += 1;
        }
        a
    };
    let x = x.max(MIN);
    let k = x.mul_add(L2E, SH);
    let r = k - SH;
    let y = r.mul_neg_add(HI, x);
    let y = r.mul_neg_add(LO, y);
    let mut q = IF[11];
    for &c in IF[..11].iter().rev() {
        q = q.mul_add(y, c);
    }
    let p = (y * y).mul_add(q, y);
    // 2^r from the low bits of k; r >= -1021 after the clamp
    let two_r: f64x4 = cast((cast::<_, i64x4>(k) + BIAS) << 52);
    (p, two_r)
}

/// [`pln_node_masses`] in `wide::f64x4`.
///
/// ### Params
///
/// * `p` - Per-call scalars
/// * `z` - Scaled Hermite nodes
/// * `w` - Weights, same length
/// * `terms` - Receives the node masses, same length
///
/// ### Returns
///
/// Sum of `terms`.
#[inline(always)]
fn node_masses_128(p: &NodeParams, z: &[f64], w: &[f64], terms: &mut [f64]) -> f64 {
    const LANES: usize = 4;
    let (vx, vem, vs, vhiv, vc2) = (
        f64x4::splat(p.xf),
        f64x4::splat(p.em),
        f64x4::splat(p.s),
        f64x4::splat(p.half_iv),
        f64x4::splat(2.0 * p.c),
    );
    let mut acc = f64x4::ZERO;
    let n = z.len() / LANES * LANES;
    for ((&zc, &wc), tc) in z
        .as_chunks::<LANES>()
        .0
        .iter()
        .zip(w.as_chunks::<LANES>().0)
        .zip(terms.as_chunks_mut::<LANES>().0)
    {
        let v = vs * f64x4::from(zc);
        let (q, two_r) = exp_parts_128(v);
        let em1 = two_r.mul_add(q, two_r - f64x4::ONE);
        let dh = vx * v - vem * em1 - v * (v + vc2) * vhiv;
        let (q, two_r) = exp_parts_128(dh);
        let ex = two_r.mul_add(q, two_r);
        let t = f64x4::from(wc) * ex;
        *tc = t.to_array();
        acc += t;
    }
    acc.reduce_add() + node_masses_tail(p, &z[n..], &w[n..], &mut terms[n..])
}

//////////
// AVX2 //
//////////

/// [`exp_parts_128`] in AVX2 with FMA.
///
/// ### Params
///
/// * `x` - Argument
///
/// ### Returns
///
/// `(expm1(y), 2^r)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn exp_parts_avx2(x: __m256d) -> (__m256d, __m256d) {
    let sh = _mm256_set1_pd(SHIFT);
    let x = _mm256_max_pd(x, _mm256_set1_pd(EXP_ARG_MIN));
    let k = _mm256_fmadd_pd(x, _mm256_set1_pd(std::f64::consts::LOG2_E), sh);
    let r = _mm256_sub_pd(k, sh);
    let y = _mm256_fnmadd_pd(r, _mm256_set1_pd(LN2_HI), x);
    let y = _mm256_fnmadd_pd(r, _mm256_set1_pd(LN2_LO), y);
    let mut q = _mm256_set1_pd(INV_FACT[11]);
    for &c in INV_FACT[..11].iter().rev() {
        q = _mm256_fmadd_pd(q, y, _mm256_set1_pd(c));
    }
    let p = _mm256_fmadd_pd(_mm256_mul_pd(y, y), q, y);
    let bits = _mm256_add_epi64(_mm256_castpd_si256(k), _mm256_set1_epi64x(EXP_BIAS));
    (p, _mm256_castsi256_pd(_mm256_slli_epi64::<52>(bits)))
}

/// [`pln_node_masses`] in AVX2 with FMA.
///
/// ### Params
///
/// * `p` - Per-call scalars
/// * `z` - Scaled Hermite nodes
/// * `w` - Weights, same length
/// * `terms` - Receives the node masses, same length
///
/// ### Returns
///
/// Sum of `terms`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn node_masses_avx2(p: &NodeParams, z: &[f64], w: &[f64], terms: &mut [f64]) -> f64 {
    const LANES: usize = 4;
    let (vx, vem, vs, vhiv, vc2, one) = (
        _mm256_set1_pd(p.xf),
        _mm256_set1_pd(p.em),
        _mm256_set1_pd(p.s),
        _mm256_set1_pd(p.half_iv),
        _mm256_set1_pd(2.0 * p.c),
        _mm256_set1_pd(1.0),
    );
    let mut acc = _mm256_setzero_pd();
    let n = z.len() / LANES * LANES;
    for i in (0..n).step_by(LANES) {
        // SAFETY: i + LANES <= n <= len of z, w and terms
        let (zc, wc) = unsafe {
            (
                _mm256_loadu_pd(z.as_ptr().add(i)),
                _mm256_loadu_pd(w.as_ptr().add(i)),
            )
        };
        let v = _mm256_mul_pd(vs, zc);
        let (q, two_r) = exp_parts_avx2(v);
        let em1 = _mm256_fmadd_pd(two_r, q, _mm256_sub_pd(two_r, one));
        let quad = _mm256_mul_pd(_mm256_mul_pd(v, _mm256_add_pd(v, vc2)), vhiv);
        let dh = _mm256_sub_pd(
            _mm256_sub_pd(_mm256_mul_pd(vx, v), _mm256_mul_pd(vem, em1)),
            quad,
        );
        let (q, two_r) = exp_parts_avx2(dh);
        let t = _mm256_mul_pd(wc, _mm256_fmadd_pd(two_r, q, two_r));
        // SAFETY: as for the loads
        unsafe { _mm256_storeu_pd(terms.as_mut_ptr().add(i), t) };
        acc = _mm256_add_pd(acc, t);
    }
    let mut lanes = [0.0; LANES];
    // SAFETY: lanes holds LANES f64
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), acc) };
    lanes.iter().sum::<f64>() + node_masses_tail(p, &z[n..], &w[n..], &mut terms[n..])
}

/////////////
// AVX-512 //
/////////////

/// [`exp_parts_128`] in AVX-512F.
///
/// ### Params
///
/// * `x` - Argument
///
/// ### Returns
///
/// `(expm1(y), 2^r)`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
fn exp_parts_avx512(x: __m512d) -> (__m512d, __m512d) {
    let sh = _mm512_set1_pd(SHIFT);
    let x = _mm512_max_pd(x, _mm512_set1_pd(EXP_ARG_MIN));
    let k = _mm512_fmadd_pd(x, _mm512_set1_pd(std::f64::consts::LOG2_E), sh);
    let r = _mm512_sub_pd(k, sh);
    let y = _mm512_fnmadd_pd(r, _mm512_set1_pd(LN2_HI), x);
    let y = _mm512_fnmadd_pd(r, _mm512_set1_pd(LN2_LO), y);
    let mut q = _mm512_set1_pd(INV_FACT[11]);
    for &c in INV_FACT[..11].iter().rev() {
        q = _mm512_fmadd_pd(q, y, _mm512_set1_pd(c));
    }
    let p = _mm512_fmadd_pd(_mm512_mul_pd(y, y), q, y);
    let bits = _mm512_add_epi64(_mm512_castpd_si512(k), _mm512_set1_epi64(EXP_BIAS));
    (p, _mm512_castsi512_pd(_mm512_slli_epi64::<52>(bits)))
}

/// [`pln_node_masses`] in AVX-512F.
///
/// ### Params
///
/// * `p` - Per-call scalars
/// * `z` - Scaled Hermite nodes
/// * `w` - Weights, same length
/// * `terms` - Receives the node masses, same length
///
/// ### Returns
///
/// Sum of `terms`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn node_masses_avx512(p: &NodeParams, z: &[f64], w: &[f64], terms: &mut [f64]) -> f64 {
    const LANES: usize = 8;
    let (vx, vem, vs, vhiv, vc2, one) = (
        _mm512_set1_pd(p.xf),
        _mm512_set1_pd(p.em),
        _mm512_set1_pd(p.s),
        _mm512_set1_pd(p.half_iv),
        _mm512_set1_pd(2.0 * p.c),
        _mm512_set1_pd(1.0),
    );
    let mut acc = _mm512_setzero_pd();
    let n = z.len() / LANES * LANES;
    for i in (0..n).step_by(LANES) {
        // SAFETY: i + LANES <= n <= len of z, w and terms
        let (zc, wc) = unsafe {
            (
                _mm512_loadu_pd(z.as_ptr().add(i)),
                _mm512_loadu_pd(w.as_ptr().add(i)),
            )
        };
        let v = _mm512_mul_pd(vs, zc);
        let (q, two_r) = exp_parts_avx512(v);
        let em1 = _mm512_fmadd_pd(two_r, q, _mm512_sub_pd(two_r, one));
        let quad = _mm512_mul_pd(_mm512_mul_pd(v, _mm512_add_pd(v, vc2)), vhiv);
        let dh = _mm512_sub_pd(
            _mm512_sub_pd(_mm512_mul_pd(vx, v), _mm512_mul_pd(vem, em1)),
            quad,
        );
        let (q, two_r) = exp_parts_avx512(dh);
        let t = _mm512_mul_pd(wc, _mm512_fmadd_pd(two_r, q, two_r));
        // SAFETY: as for the loads
        unsafe { _mm512_storeu_pd(terms.as_mut_ptr().add(i), t) };
        acc = _mm512_add_pd(acc, t);
    }
    let mut lanes = [0.0; LANES];
    // SAFETY: lanes holds LANES f64
    unsafe { _mm512_storeu_pd(lanes.as_mut_ptr(), acc) };
    lanes.iter().sum::<f64>() + node_masses_tail(p, &z[n..], &w[n..], &mut terms[n..])
}
