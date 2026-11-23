//! The one entry point: [`run_cnv`] over a CSR count matrix.
//!
//! Inputs arrive as already-validated `u32` / `u64` numpy arrays (the Python
//! layer does the dtype coercion and chromosome parsing) and are borrowed, not
//! copied. The result goes back as a flat dictionary of numpy arrays; the
//! Python layer scatters per-used-cell rows to full length and builds the
//! tables.

use cnv_sc_rs::cnv::{CnvResult, run_cnv as core_run_cnv};
use cnv_sc_rs::genome::CsrCounts;
use cnv_sc_rs::posterior::{N_LOG_ODDS, effective_state};
use cnv_sc_rs::prelude::Verbosity;
use cnv_sc_rs::reference::Reference;
use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::CnvErr;
use crate::params::cnv_params;

/////////////
// Helpers //
/////////////

/// Borrow a contiguous 1-D array.
///
/// ### Params
///
/// * `a` - The array
/// * `name` - Argument name, for the error
///
/// ### Returns
///
/// The slice, or a `ValueError` if the array is not contiguous.
fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))
}

/// Lower-case `Debug` name of a state enum, e.g. `Amp1` -> `"amp1"`.
///
/// ### Params
///
/// * `s` - The state
///
/// ### Returns
///
/// The name as numbat spells it.
fn state_name(s: impl std::fmt::Debug) -> String {
    format!("{s:?}").to_lowercase()
}

/// Flatten a [`CnvResult`] into a dictionary of numpy arrays and lists.
///
/// ### Params
///
/// * `py` - The interpreter
/// * `r` - The core's result
///
/// ### Returns
///
/// The dictionary. Keys are documented in `cnv_sc_rs/_cnv_sc_rs.pyi`.
fn pack<'py>(py: Python<'py>, r: CnvResult<f32>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let n_cells = r.bins.n_cells;
    d.set_item(
        "refs",
        r.refs
            .iter()
            .map(|x| x.map_or(-1, |r| r as i32))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    // bins
    let n_bins = r.bins.bins.len();
    let o = &r.order;
    let gene = |k: u32| r.bins.kept[k as usize] as usize;
    d.set_item(
        "bins",
        r.bins.data.into_pyarray(py).reshape([n_cells, n_bins])?,
    )?;
    let bins = &r.bins.bins;
    d.set_item(
        "bin_chrom",
        bins.iter()
            .map(|b| b.chrom)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "bin_start",
        bins.iter()
            .map(|b| o.start[gene(b.lo)])
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "bin_end",
        bins.iter()
            .map(|b| o.end[gene(b.hi - 1)])
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "bin_n_genes",
        bins.iter()
            .map(|b| b.hi - b.lo)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;

    // cells and leaf clusters
    d.set_item("cells", r.cells.clone().into_pyarray(py))?;
    let mut leaf = vec![-1i32; n_cells];
    for g in &r.groups {
        if let Some(id) = g.leaf {
            for &c in &g.cells {
                leaf[c as usize] = id as i32;
            }
        }
    }
    d.set_item("leaf", leaf.into_pyarray(py))?;

    // consensus segments
    let segs = &r.consensus;
    let col_u32 =
        |f: fn(&cnv_sc_rs::segments::Segment) -> u32| segs.iter().map(f).collect::<Vec<_>>();
    d.set_item("seg_chrom", col_u32(|s| s.chrom).into_pyarray(py))?;
    d.set_item("seg_seg", col_u32(|s| s.seg).into_pyarray(py))?;
    d.set_item("seg_n_genes", col_u32(|s| s.n_genes).into_pyarray(py))?;
    d.set_item(
        "seg_start",
        segs.iter()
            .map(|s| s.start)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "seg_end",
        segs.iter()
            .map(|s| s.end)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "seg_bulk",
        segs.iter()
            .map(|s| s.bulk.map_or(-1, |b| b as i32))
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    d.set_item(
        "seg_state",
        segs.iter().map(|s| state_name(s.state)).collect::<Vec<_>>(),
    )?;
    d.set_item(
        "seg_state_post",
        segs.iter()
            .map(|s| state_name(effective_state(s)))
            .collect::<Vec<_>>(),
    )?;
    let retest = |f: fn(&cnv_sc_rs::segments::Retest) -> f64| {
        segs.iter()
            .map(|s| s.retest.as_ref().map_or(f64::NAN, f))
            .collect::<Vec<_>>()
    };
    d.set_item("seg_phi_hat", retest(|t| t.phi_hat).into_pyarray(py))?;
    d.set_item("seg_s_phi", retest(|t| t.s_phi).into_pyarray(py))?;
    d.set_item("seg_llr", retest(|t| t.llr).into_pyarray(py))?;
    d.set_item("seg_p_neu", retest(|t| t.p_neu).into_pyarray(py))?;
    d.set_item("seg_p_amp", retest(|t| t.p_amp).into_pyarray(py))?;
    d.set_item("seg_p_del", retest(|t| t.p_del).into_pyarray(py))?;
    d.set_item("seg_p_loh", retest(|t| t.p_loh).into_pyarray(py))?;
    let multi = PyList::empty(py);
    for m in &r.multi_state {
        match m {
            Some(states) => {
                multi.append(states.iter().map(|&s| state_name(s)).collect::<Vec<_>>())?
            }
            None => multi.append(py.None())?,
        }
    }
    d.set_item("seg_multi_state", multi)?;

    // per-cell posteriors and clones; absent when nothing is non-neutral
    if let Some(p) = r.posteriors {
        let (n_used, n_post) = (p.cells.len(), p.segs.len());
        d.set_item(
            "post_segs",
            p.segs
                .iter()
                .map(|&i| i as u32)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        d.set_item("entropy", r.entropy.into_pyarray(py))?;
        d.set_item("confident", r.confident.into_pyarray(py))?;
        d.set_item(
            "log_odds",
            p.log_odds
                .into_pyarray(py)
                .reshape([n_used, n_post, N_LOG_ODDS])?,
        )?;
        d.set_item("mu", p.mu.into_pyarray(py))?;
        d.set_item("sigma", p.sigma.into_pyarray(py))?;
    }
    if let Some(c) = r.clones {
        let n_geno = c.genotypes.len();
        let n_used = c.clone.len();
        d.set_item(
            "clone_prob",
            c.prob.into_pyarray(py).reshape([n_used, n_geno])?,
        )?;
        d.set_item("clone", c.clone.into_pyarray(py))?;
        d.set_item("p_tumour", c.p_tumour.into_pyarray(py))?;
        let genos = PyList::empty(py);
        for g in &c.genotypes {
            let events: Vec<(usize, Option<String>)> =
                g.iter().map(|e| (e.col, e.state.map(state_name))).collect();
            genos.append(events)?;
        }
        d.set_item("genotypes", genos)?;
    }
    Ok(d)
}

/////////////////
// Entry point //
/////////////////

/// Expression-only CNV inference over a cells x genes CSR count matrix.
///
/// Runs the whole tier 0 pipeline with the GIL released, in a pool of
/// `n_threads` when given, rayon's global pool otherwise.
///
/// ### Params
///
/// * `indptr` - CSR row offsets, `u32`, length `n_cells + 1`
/// * `indices` - CSR gene index per nonzero, `u32`
/// * `data` - CSR count per nonzero, `u32`
/// * `n_genes` - Number of genes (columns)
/// * `chrom` - Chromosome id per gene, `u32`
/// * `start` - Gene start per gene, `u64`
/// * `end` - Gene end per gene, `u64`
/// * `params` - Parameters, see [`cnv_params`]
/// * `ref_cells` - Row indices of the normal reference cells, `u32`
/// * `ref_group` - Profile per entry of `ref_cells`, `u32`, labels
///   `0..n_ref`; `None` pools all reference cells into one profile
/// * `ref_profiles` - Precomputed profiles, `f32`, `n_ref x n_genes`
///   C-contiguous; exclusive with `ref_cells`
/// * `n_threads` - Thread count, `None` for rayon's global pool
/// * `verbose` - Print stage timings to stdout
///
/// ### Returns
///
/// The flattened result, or `CnvError` / `ValueError`.
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, n_genes, chrom, start, end, params, *, ref_cells = None, ref_group = None, ref_profiles = None, n_threads = None, verbose = false))]
#[allow(clippy::too_many_arguments)]
pub fn run_cnv<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, u32>,
    indices: PyReadonlyArray1<'py, u32>,
    data: PyReadonlyArray1<'py, u32>,
    n_genes: usize,
    chrom: PyReadonlyArray1<'py, u32>,
    start: PyReadonlyArray1<'py, u64>,
    end: PyReadonlyArray1<'py, u64>,
    params: &Bound<'py, PyDict>,
    ref_cells: Option<PyReadonlyArray1<'py, u32>>,
    ref_group: Option<PyReadonlyArray1<'py, u32>>,
    ref_profiles: Option<PyReadonlyArray2<'py, f32>>,
    n_threads: Option<usize>,
    verbose: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let params = cnv_params(params)?;
    let (indptr, indices, data) = (
        slice(&indptr, "indptr")?,
        slice(&indices, "indices")?,
        slice(&data, "data")?,
    );
    let (chrom, start, end) = (
        slice(&chrom, "chrom")?,
        slice(&start, "start")?,
        slice(&end, "end")?,
    );
    let reference = match (&ref_cells, &ref_group, &ref_profiles) {
        (Some(c), None, None) => Reference::Cells(slice(c, "ref_cells")?),
        (Some(c), Some(g), None) => {
            let group = slice(g, "ref_group")?;
            Reference::Groups {
                cells: slice(c, "ref_cells")?,
                group,
                n_ref: group.iter().max().map_or(0, |&m| m as usize + 1),
            }
        }
        (None, None, Some(p)) => {
            if !p.is_c_contiguous() {
                return Err(PyValueError::new_err("ref_profiles must be C-contiguous"));
            }
            Reference::Profiles {
                lambda: p.as_slice()?,
                n_ref: p.shape()[0],
            }
        }
        _ => {
            return Err(PyValueError::new_err(
                "pass ref_cells (optionally with ref_group) or ref_profiles",
            ));
        }
    };
    let verbosity = if verbose {
        Verbosity::Normal
    } else {
        Verbosity::Quiet
    };
    let pool = match n_threads {
        Some(n) => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyValueError::new_err(format!("could not build thread pool: {e}")))?,
        ),
        None => None,
    };

    let result = py
        .detach(|| {
            let work = || {
                let counts = CsrCounts::new(indptr, indices, data, n_genes)?;
                core_run_cnv::<f32>(
                    &counts,
                    chrom,
                    start,
                    end,
                    &reference,
                    Some(params),
                    verbosity,
                )
            };
            match &pool {
                Some(p) => p.install(work),
                None => work(),
            }
        })
        .map_err(CnvErr)?;
    pack(py, result)
}
