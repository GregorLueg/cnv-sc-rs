//! Copy number inference from single cell expression.
//!
//! A port of the expression path of numbat (Gao et al., *Nature Biotechnology*
//! 2023) and its numerics package hahmmr, both MIT. Tier 0: counts, gene
//! coordinates and a reference give an inferCNV-style smoothed signal, a cell
//! hierarchy, a Poisson-lognormal HMM on pseudobulks, consensus segments,
//! per-cell segment posteriors and a tree-free tumour/normal call. See
//! `docs/SPEC.md` and `PROVENANCE.md`.
//!
//! ### Inputs
//!
//! Plain slices only: a cells x genes CSR count matrix with `u32` indices and
//! counts, per-gene chromosome and position, and a reference (normal cells or
//! profiles). No I/O and no store types; adapters live downstream.
//!
//! ### Numeric policy
//!
//! Storage is generic over [`utils::traits::CnvFloat`], default `f32`. Every
//! reduction and every loglikelihood accumulates in `f64`: segment
//! loglikelihoods are sums over thousands of genes whose interesting
//! differences are `O(1)`.

#![warn(missing_docs)]
// Indexed loops are what the numeric kernels want.
#![allow(clippy::needless_range_loop)]

/// Version of this crate, for bindings that vendor it and need to say which.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod cluster;
pub mod errors;
pub mod genome;
pub mod hmm;
pub mod model;
pub mod prelude;
pub mod reference;
pub mod segments;
pub mod smooth;
pub mod utils;
