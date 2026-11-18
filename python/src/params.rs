//! [`CnvParams`] from a Python dictionary.
//!
//! A missing key keeps the crate's `Default`, so the Rust side stays the source
//! of truth. An unknown key is an error naming the key: a silently ignored typo
//! in a stringly-typed dictionary costs somebody an afternoon.

use cnv_sc_rs::cnv::CnvParams;
use cnv_sc_rs::genome::{EXCLUDE_HG19, EXCLUDE_HG38};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Build [`CnvParams`] from `dict`, starting from the crate defaults.
///
/// Values are not range-checked here; the core validates them on use.
///
/// ### Params
///
/// * `dict` - Parameter name to value. Keys are `exclude` (`"hg38"`, `"hg19"`
///   or `"none"`), `window`, `cap`, `bin_size`, `init_k`, `min_cells`, `t`,
///   `logphi_min`, `min_genes`, `min_llr`, `min_overlap`, `max_entropy`,
///   `p_multi`
///
/// ### Returns
///
/// The parameters, or a `ValueError` / `TypeError` naming the bad key.
pub(crate) fn cnv_params(dict: &Bound<'_, PyDict>) -> PyResult<CnvParams> {
    let mut p = CnvParams::default();
    for (k, v) in dict.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "exclude" => {
                let build: String = v.extract()?;
                p.exclude = match build.as_str() {
                    "hg38" => EXCLUDE_HG38.to_vec(),
                    "hg19" => EXCLUDE_HG19.to_vec(),
                    "none" => Vec::new(),
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "exclude must be 'hg38', 'hg19' or 'none', got {other:?}"
                        )));
                    }
                };
            }
            "window" => p.smooth.window = v.extract()?,
            "cap" => p.smooth.cap = v.extract()?,
            "bin_size" => p.smooth.bin_size = v.extract()?,
            "init_k" => p.init_k = v.extract()?,
            "min_cells" => p.min_cells = v.extract()?,
            "t" => p.hmm.t = v.extract()?,
            "logphi_min" => p.hmm.logphi_min = v.extract()?,
            "min_genes" => p.min_genes = v.extract()?,
            "min_llr" => p.consensus.min_llr = v.extract()?,
            "min_overlap" => p.consensus.min_overlap = v.extract()?,
            "max_entropy" => p.max_entropy = v.extract()?,
            "p_multi" => p.p_multi = v.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown parameter {other:?}"
                )));
            }
        }
    }
    Ok(p)
}

/// The crate defaults as a dictionary, so the Python layer can be tested
/// against them.
///
/// ### Params
///
/// * `py` - The interpreter
///
/// ### Returns
///
/// Every key [`cnv_params`] accepts except `exclude`, with its default.
#[pyfunction]
pub fn default_params(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let p = CnvParams::default();
    let d = PyDict::new(py);
    d.set_item("window", p.smooth.window)?;
    d.set_item("cap", p.smooth.cap)?;
    d.set_item("bin_size", p.smooth.bin_size)?;
    d.set_item("init_k", p.init_k)?;
    d.set_item("min_cells", p.min_cells)?;
    d.set_item("t", p.hmm.t)?;
    d.set_item("logphi_min", p.hmm.logphi_min)?;
    d.set_item("min_genes", p.min_genes)?;
    d.set_item("min_llr", p.consensus.min_llr)?;
    d.set_item("min_overlap", p.consensus.min_overlap)?;
    d.set_item("max_entropy", p.max_entropy)?;
    d.set_item("p_multi", p.p_multi)?;
    Ok(d)
}
