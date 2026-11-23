//! Bounded 1-D maximisation for concave objectives.

use crate::errors::CnvErrors;

/// Safeguarded Newton for the maximum of a concave function on `[lo, hi]`.
///
/// Works on the derivative only: `f'` is non-increasing, so every evaluation
/// shrinks a bracket around the root of `f'`. Newton steps that leave the
/// bracket fall back to bisection. An unevaluated bound is tried once before
/// bisecting towards it, so a maximum on the boundary is found in a few steps.
///
/// ### Params
///
/// * `deriv` - Returns `(f'(x), f''(x))`
/// * `lo` - Lower bound
/// * `hi` - Upper bound
/// * `start` - Starting point, clamped to `[lo, hi]`
/// * `tol` - Stop once a step or the bracket is below `tol * (1 + |x|)`
/// * `max_iter` - Iteration cap
/// * `what` - Name for the error message
///
/// ### Returns
///
/// The maximiser, or [`CnvErrors::NonConvergence`].
pub fn maximise_concave_1d<F>(
    mut deriv: F,
    lo: f64,
    hi: f64,
    start: f64,
    tol: f64,
    max_iter: usize,
    what: &'static str,
) -> Result<f64, CnvErrors>
where
    F: FnMut(f64) -> (f64, f64),
{
    let (mut a, mut b) = (lo, hi);
    let (mut lo_tried, mut hi_tried) = (false, false);
    let mut x = start.clamp(lo, hi);
    for _ in 0..max_iter {
        lo_tried |= x == lo;
        hi_tried |= x == hi;
        let (g, h) = deriv(x);
        if g == 0.0 {
            return Ok(x);
        }
        if g > 0.0 {
            if x == hi {
                return Ok(hi);
            }
            a = x;
        } else {
            if x == lo {
                return Ok(lo);
            }
            b = x;
        }
        let newton = x - g / h;
        let next = if h < 0.0 && newton > a && newton < b {
            newton
        } else if h < 0.0 && newton >= b && b == hi && !hi_tried {
            hi
        } else if h < 0.0 && newton <= a && a == lo && !lo_tried {
            lo
        } else {
            0.5 * (a + b)
        };
        let scale = tol * (1.0 + next.abs());
        if (next - x).abs() <= scale || b - a <= scale {
            return Ok(next);
        }
        x = next;
    }
    Err(CnvErrors::NonConvergence {
        what,
        iterations: max_iter,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `f(x) = -(x - c)^2`
    fn quad(c: f64) -> impl FnMut(f64) -> (f64, f64) {
        move |x| (-2.0 * (x - c), -2.0)
    }

    #[test]
    fn test_maximise_concave_1d_interior() {
        let x = maximise_concave_1d(quad(2.5), 0.1, 10.0, 1.0, 1e-12, 50, "q").unwrap();
        assert_relative_eq!(x, 2.5, max_relative = 1e-12);
    }

    #[test]
    fn test_maximise_concave_1d_boundaries() {
        let x = maximise_concave_1d(quad(20.0), 0.1, 10.0, 1.0, 1e-12, 50, "q").unwrap();
        assert_eq!(x, 10.0);
        let x = maximise_concave_1d(quad(-3.0), 0.1, 10.0, 1.0, 1e-12, 50, "q").unwrap();
        assert_eq!(x, 0.1);
    }

    #[test]
    fn test_maximise_concave_1d_non_quadratic() {
        // f(x) = ln x - x, max at 1
        let x = maximise_concave_1d(
            |x| (1.0 / x - 1.0, -1.0 / (x * x)),
            0.01,
            100.0,
            50.0,
            1e-12,
            100,
            "log",
        )
        .unwrap();
        assert_relative_eq!(x, 1.0, max_relative = 1e-10);
    }
}
