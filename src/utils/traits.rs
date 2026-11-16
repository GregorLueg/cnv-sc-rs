//! Shared trait bounds.

use bonsai_rs::utils::traits::BonsaiFloat;

/// Floating-point types `cnv-sc-rs` stores matrices in.
///
/// A superset of [`BonsaiFloat`] so binned signal can go straight into
/// `bonsai_rs::tree::linkage::linkage_tree`. Governs storage only: every
/// reduction accumulates in `f64` (see [`wide`] and [`narrow`]).
pub trait CnvFloat: BonsaiFloat {}

impl<T> CnvFloat for T where T: BonsaiFloat {}

/// Widen a storage float to `f64` for accumulation.
///
/// ### Params
///
/// * `x` - Value to widen
///
/// ### Returns
///
/// The value as an `f64`; `NAN` is unreachable for `f32` and `f64`.
#[inline(always)]
pub fn wide<T: CnvFloat>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}

/// Narrow an `f64` back to the storage float.
///
/// ### Params
///
/// * `x` - Value to narrow
///
/// ### Returns
///
/// The value as a `T`.
#[inline(always)]
pub fn narrow<T: CnvFloat>(x: f64) -> T {
    T::from_f64(x).unwrap_or_else(T::nan)
}
