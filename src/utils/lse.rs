//! Log-space helpers shared by the HMM, the retest and the posteriors.

/// Log of a sum of exponentials, max-shifted.
///
/// Matches hahmmr's `logSumExp` (src/hmm.cpp:20): `-inf` terms are skipped and
/// an all `-inf` input returns `-inf`.
///
/// ### Params
///
/// * `x` - Log-space terms
///
/// ### Returns
///
/// `ln(sum(exp(x)))`.
#[inline]
pub fn log_sum_exp(x: &[f64]) -> f64 {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    let s: f64 = x
        .iter()
        .filter(|v| **v > f64::NEG_INFINITY)
        .map(|v| (v - m).exp())
        .sum();
    m + s.ln()
}

/// `ln(1 - exp(-a))` for `a >= 0`, accurate across the range.
///
/// Mächler's split at `ln 2`: `ln(-expm1(-a))` for small `a`,
/// `ln1p(-exp(-a))` for large.
///
/// ### Params
///
/// * `a` - Non-negative argument
///
/// ### Returns
///
/// `ln(1 - exp(-a))`; `-inf` at `a = 0`.
///
/// ### References
///
/// Mächler, *Accurately computing log(1 - exp(-|a|))*, Rmpfr vignette, 2012.
#[inline]
pub fn log1mexp(a: f64) -> f64 {
    if a <= std::f64::consts::LN_2 {
        (-(-a).exp_m1()).ln()
    } else {
        (-(-a).exp()).ln_1p()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_log_sum_exp_matches_naive() {
        let x = [-1.0, 0.5, 2.0];
        let naive = x.iter().map(|v: &f64| v.exp()).sum::<f64>().ln();
        assert_relative_eq!(log_sum_exp(&x), naive, max_relative = 1e-14);
    }

    #[test]
    fn test_log_sum_exp_skips_neg_inf() {
        assert_relative_eq!(log_sum_exp(&[f64::NEG_INFINITY, 0.0]), 0.0);
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY; 3]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp_large_values() {
        assert_relative_eq!(
            log_sum_exp(&[1000.0, 1000.0]),
            1000.0 + std::f64::consts::LN_2,
            max_relative = 1e-14
        );
    }

    #[test]
    fn test_log1mexp_both_branches() {
        for a in [1e-10_f64, 0.1, 0.69, 0.7, 5.0, 40.0] {
            let naive = (1.0 - (-a).exp()).ln();
            assert_relative_eq!(log1mexp(a), naive, max_relative = 1e-6);
        }
    }
}
