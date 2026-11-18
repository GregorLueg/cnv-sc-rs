//! Rust errors as Python exceptions.
//!
//! The mapping is flat: every [`CnvErrors`] becomes a `CnvError` carrying the
//! core's message. Most of what a caller can get wrong by hand (dtype, shape,
//! chromosome labels) is caught in `cnv_sc_rs._validate` first, so what comes
//! back from here is usually a data problem the core detected (too few cells,
//! no signal, malformed CSR).

use cnv_sc_rs::errors::CnvErrors;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    _cnv_sc_rs,
    CnvError,
    PyException,
    "Raised when the cnv-sc-rs Rust core fails."
);

/// Newtype carrying a [`CnvErrors`] across the orphan rule.
///
/// `.map_err(CnvErr)?` in a `PyResult` function.
pub(crate) struct CnvErr(pub CnvErrors);

impl From<CnvErr> for PyErr {
    fn from(e: CnvErr) -> PyErr {
        CnvError::new_err(e.0.to_string())
    }
}
