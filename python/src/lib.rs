//! Python bindings for `cnv-sc-rs`.
//!
//! Deliberately thin: one entry point, [`run::run_cnv`], taking a validated CSR
//! matrix, gene loci, the reference cells and a parameter dictionary. Input
//! coercion, chromosome parsing, result tables and the AnnData layer live in
//! the hand-written `cnv_sc_rs` Python package on top.
//!
//! The crate owns the defaults: a parameter absent from the dictionary keeps
//! `CnvParams::default()`. Storage is `f32`, the core's default.

#![warn(missing_docs)]

use pyo3::prelude::*;

mod error;
mod params;
mod run;

/// Assemble the extension module.
///
/// ### Params
///
/// * `m` - The module being initialised
///
/// ### Returns
///
/// Nothing, or the first registration error.
#[pymodule]
fn _cnv_sc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__core_version__", cnv_sc_rs::VERSION)?;
    m.add("CnvError", m.py().get_type::<error::CnvError>())?;
    m.add_function(wrap_pyfunction!(run::run_cnv, m)?)?;
    m.add_function(wrap_pyfunction!(params::default_params, m)?)?;
    Ok(())
}
