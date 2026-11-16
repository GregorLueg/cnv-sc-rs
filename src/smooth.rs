//! inferCNV-style smoothed log-ratio, reduced on the fly to genomic bins (SPEC 3).
//!
//! The dense cells x genes matrix never exists. A zero count gives the same
//! value `z_g` in every cell, and centring, running mean and bin averaging
//! are linear, so a cell's bins are a precomputed baseline plus the spread
//! of its nonzeros through fixed gene-to-bin weights (SPEC 3.3). Cost per
//! cell is O(nnz * bins a window touches + n_bins).

use rayon::prelude::*;

use crate::errors::CnvErrors;
use crate::genome::{CsrCounts, GenomeOrder};
use crate::utils::traits::{CnvFloat, narrow};

/// Scale from proportions to counts per million (N/R/utils.R:113).
const CPM: f64 = 1e6;

/// Minimum cells per rayon task. Ours, not swept: a cell costs about 5 us
/// on one thread at G = 10k, 5% density (benches/smooth.rs, 2026-11-16).
const CELL_MIN_LEN: usize = 64;

//////////////////
// SmoothParams //
//////////////////

/// Smoothing and binning parameters (SPEC 3).
#[derive(Clone, Copy, Debug)]
pub struct SmoothParams {
    /// Running-mean window in kept genes, `window` in N/R/utils.R:103
    pub window: usize,
    /// Symmetric cap on the log-ratio, `cap` in N/R/utils.R:116
    pub cap: f64,
    /// Kept genes per bin (SPEC 3.2, ours)
    pub bin_size: usize,
}

impl SmoothParams {
    /// Create parameters.
    ///
    /// ### Params
    ///
    /// * `window` - Running-mean window, `>= 1`
    /// * `cap` - Log-ratio cap, `> 0`
    /// * `bin_size` - Kept genes per bin, `>= 1`
    ///
    /// ### Returns
    ///
    /// The parameters (validated on use).
    pub fn new(window: usize, cap: f64, bin_size: usize) -> Self {
        Self {
            window,
            cap,
            bin_size,
        }
    }

    /// Check the ranges.
    ///
    /// ### Returns
    ///
    /// `Ok`, or [`CnvErrors::InvalidParameter`].
    fn validate(&self) -> Result<(), CnvErrors> {
        if self.window == 0 {
            return Err(CnvErrors::InvalidParameter {
                name: "window",
                value: self.window.to_string(),
                expected: ">= 1",
            });
        }
        if !(self.cap > 0.0 && self.cap.is_finite()) {
            return Err(CnvErrors::InvalidParameter {
                name: "cap",
                value: self.cap.to_string(),
                expected: "finite and > 0",
            });
        }
        if self.bin_size == 0 {
            return Err(CnvErrors::InvalidParameter {
                name: "bin_size",
                value: self.bin_size.to_string(),
                expected: ">= 1",
            });
        }
        Ok(())
    }
}

impl Default for SmoothParams {
    /// numbat's `window = 101`, `cap = 3` (N/R/utils.R:103, 116) and our
    /// `bin_size = 100`.
    fn default() -> Self {
        Self::new(101, 3.0, 100)
    }
}

/////////
// Bin //
/////////

/// One bin: kept genes `[lo, hi)` (indices into [`SmoothedBins::kept`]),
/// all on one chromosome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bin {
    /// Chromosome id
    pub chrom: u32,
    /// First kept gene
    pub lo: u32,
    /// One past the last kept gene
    pub hi: u32,
}

/// Smoothed signal per cell and bin.
#[derive(Clone, Debug)]
pub struct SmoothedBins<T: CnvFloat> {
    /// Cells x bins, row-major
    pub data: Vec<T>,
    /// Number of cells (rows), every input cell
    pub n_cells: usize,
    /// Bin layout, genomic order
    pub bins: Vec<Bin>,
    /// Genomic indices of the kept genes, ascending
    pub kept: Vec<u32>,
}

/// Bins over kept genes (SPEC 3.2): runs of `bin_size` per chromosome, the
/// remainder its own bin unless below `bin_size / 2`, then merged into the
/// previous one.
///
/// ### Params
///
/// * `runs` - `(chrom, lo, hi)` per chromosome, kept-gene indices
/// * `bin_size` - Kept genes per bin
///
/// ### Returns
///
/// The bins, genomic order.
fn bin_layout(runs: &[(u32, usize, usize)], bin_size: usize) -> Vec<Bin> {
    let mut bins = Vec::new();
    for &(chrom, lo, hi) in runs {
        let first = bins.len();
        let mut s = lo;
        while s < hi {
            let e = (s + bin_size).min(hi);
            if bins.len() > first && 2 * (e - s) < bin_size {
                let last: &mut Bin = bins.last_mut().expect("bin exists");
                last.hi = e as u32;
            } else {
                bins.push(Bin {
                    chrom,
                    lo: s as u32,
                    hi: e as u32,
                });
            }
            s = e;
        }
    }
    bins
}

/// Running-mean window of position `i` in a run of `n` (SPEC 3.1).
///
/// caTools `runmean(align = "center", endrule = "mean")`: `(k-1)/2` genes
/// left and `k/2` right, truncated at the run ends.
///
/// ### Params
///
/// * `i` - Position in the run
/// * `n` - Run length
/// * `k` - Window
///
/// ### Returns
///
/// `[lo, hi)` within the run.
#[inline]
fn window_at(i: usize, n: usize, k: usize) -> (usize, usize) {
    (i.saturating_sub((k - 1) / 2), (i + k / 2 + 1).min(n))
}

//////////
// Plan //
//////////

/// Everything about the transform that does not depend on the cell.
struct Plan {
    /// Kept genes, genomic indices
    kept: Vec<u32>,
    /// Caller gene index to kept index, `u32::MAX` if not kept
    kept_of: Vec<u32>,
    /// `log2(lambda_bar*1e6 + 1)` per kept gene
    r: Vec<f64>,
    /// `clamp(-r, -cap, cap)`: the value of a zero count, per kept gene
    z: Vec<f64>,
    /// Bin layout
    bins: Vec<Bin>,
    /// Weight offsets per kept gene into `w_val`, length `n_kept + 1`
    w_ptr: Vec<u32>,
    /// First bin each kept gene's weights land in
    w_first: Vec<u32>,
    /// Weights `(B S e_g)_b` for consecutive bins from `w_first[g]`
    w_val: Vec<f64>,
    /// `B S z - mean(z)` per bin
    base: Vec<f64>,
}

impl Plan {
    /// Build the plan.
    ///
    /// ### Params
    ///
    /// * `order` - Genomic order
    /// * `keep` - Kept genes, genomic order
    /// * `lambda_bar` - Reference, genomic order
    /// * `params` - Validated parameters
    ///
    /// ### Returns
    ///
    /// The plan, or an error if no gene is kept.
    fn new(
        order: &GenomeOrder,
        keep: &[bool],
        lambda_bar: &[f64],
        params: &SmoothParams,
    ) -> Result<Self, CnvErrors> {
        let kept: Vec<u32> = (0..order.n_genes() as u32)
            .filter(|&i| keep[i as usize])
            .collect();
        let nk = kept.len();
        if nk == 0 {
            return Err(CnvErrors::EmptyInput { what: "kept genes" });
        }
        let mut kept_of = vec![u32::MAX; order.n_genes()];
        for (k, &i) in kept.iter().enumerate() {
            kept_of[order.perm[i as usize] as usize] = k as u32;
        }
        let r: Vec<f64> = kept
            .iter()
            .map(|&i| (lambda_bar[i as usize] * CPM + 1.0).log2())
            .collect();
        let z: Vec<f64> = r
            .iter()
            .map(|&v| (-v).clamp(-params.cap, params.cap))
            .collect();

        let mut runs: Vec<(u32, usize, usize)> = Vec::new();
        for (k, &i) in kept.iter().enumerate() {
            let c = order.chrom[i as usize];
            match runs.last_mut() {
                Some(run) if run.0 == c => run.2 = k + 1,
                _ => runs.push((c, k, k + 1)),
            }
        }
        let bins = bin_layout(&runs, params.bin_size);
        let mut bin_of = vec![0u32; nk];
        for (b, bin) in bins.iter().enumerate() {
            bin_of[bin.lo as usize..bin.hi as usize].fill(b as u32);
        }

        // Gene g at run position j feeds outputs i whose window holds j:
        // i in [j - k/2, j + (k-1)/2]. Each adds 1/|W_i| / |bin(i)|.
        let k = params.window;
        let mut w_ptr = Vec::with_capacity(nk + 1);
        let mut w_first = Vec::with_capacity(nk);
        let mut w_val: Vec<f64> = Vec::new();
        w_ptr.push(0u32);
        for &(_, lo, hi) in &runs {
            let n = hi - lo;
            for j in 0..n {
                let (i_lo, i_hi) = (j.saturating_sub(k / 2), (j + (k - 1) / 2 + 1).min(n));
                let first = bin_of[lo + i_lo];
                let start = w_val.len();
                w_val.resize(start + (bin_of[lo + i_hi - 1] - first) as usize + 1, 0.0);
                for i in i_lo..i_hi {
                    let (a, b) = window_at(i, n, k);
                    let bin = &bins[bin_of[lo + i] as usize];
                    w_val[start + (bin_of[lo + i] - first) as usize] +=
                        1.0 / ((b - a) as f64 * (bin.hi - bin.lo) as f64);
                }
                w_first.push(first);
                w_ptr.push(w_val.len() as u32);
            }
        }

        let mut base = vec![0.0f64; bins.len()];
        for g in 0..nk {
            let w = &w_val[w_ptr[g] as usize..w_ptr[g + 1] as usize];
            for (t, &v) in w.iter().enumerate() {
                base[w_first[g] as usize + t] += z[g] * v;
            }
        }
        let mean_z = z.iter().sum::<f64>() / nk as f64;
        base.iter_mut().for_each(|b| *b -= mean_z);

        Ok(Self {
            kept,
            kept_of,
            r,
            z,
            bins,
            w_ptr,
            w_first,
            w_val,
            base,
        })
    }

    /// Bins of one cell (SPEC 3.3).
    ///
    /// ### Params
    ///
    /// * `idx` - The cell's gene indices, caller order
    /// * `val` - The cell's counts
    /// * `cap` - Log-ratio cap
    /// * `acc` - Output, one per bin, overwritten
    fn cell(&self, idx: &[u32], val: &[u32], cap: f64, acc: &mut [f64]) {
        acc.copy_from_slice(&self.base);
        let lib: u64 = idx
            .iter()
            .zip(val)
            .filter(|&(&j, _)| self.kept_of[j as usize] != u32::MAX)
            .map(|(_, &y)| y as u64)
            .sum();
        if lib == 0 {
            return;
        }
        let scale = CPM / lib as f64;
        let mut sum_d = 0.0;
        for (&j, &y) in idx.iter().zip(val) {
            let g = self.kept_of[j as usize];
            if g == u32::MAX {
                continue;
            }
            let g = g as usize;
            let a = (y as f64 * scale + 1.0).log2() - self.r[g];
            let d = a.clamp(-cap, cap) - self.z[g];
            sum_d += d;
            let w = &self.w_val[self.w_ptr[g] as usize..self.w_ptr[g + 1] as usize];
            let out = &mut acc[self.w_first[g] as usize..][..w.len()];
            for (o, &v) in out.iter_mut().zip(w) {
                *o += d * v;
            }
        }
        let shift = sum_d / self.kept.len() as f64;
        acc.iter_mut().for_each(|o| *o -= shift);
    }
}

/////////////////
// smooth_bins //
/////////////////

/// Smoothed log-ratio averaged over bins, for every cell (SPEC 3).
///
/// Per cell: `x = Y / sum_{kept} Y`, `E = clamp(log2(x*1e6+1) -
/// log2(lambda_bar*1e6+1), -cap, cap)`, centred over kept genes, running
/// mean per chromosome (Deviation D1), averaged over bins (Deviation D2).
/// Evaluated sparsely (SPEC 3.3), parallel over cells, accumulated in `f64`.
/// A cell with no counts in the kept genes gets the all-zero-count profile.
///
/// ### Params
///
/// * `counts` - Count matrix
/// * `order` - Genomic order of the genes
/// * `keep` - Kept genes from [`crate::reference::filter_genes`], genomic order
/// * `lambda_bar` - Reference from [`crate::reference::lambdas_bar`], genomic order
/// * `params` - `None` for [`SmoothParams::default`]
///
/// ### Returns
///
/// Cells x bins, or an error on mismatched shapes, bad parameters or no
/// kept gene.
pub fn smooth_bins<T: CnvFloat>(
    counts: &CsrCounts,
    order: &GenomeOrder,
    keep: &[bool],
    lambda_bar: &[f64],
    params: Option<SmoothParams>,
) -> Result<SmoothedBins<T>, CnvErrors> {
    let params = params.unwrap_or_default();
    params.validate()?;
    let n = order.n_genes();
    for (what, len) in [
        ("genes in counts", counts.n_genes),
        ("keep", keep.len()),
        ("lambda_bar", lambda_bar.len()),
    ] {
        if len != n {
            return Err(CnvErrors::ShapeMismatch {
                what,
                expected: n,
                got: len,
            });
        }
    }
    let plan = Plan::new(order, keep, lambda_bar, &params)?;
    let nb = plan.bins.len();
    let n_cells = counts.n_cells();
    let mut data = vec![T::zero(); n_cells * nb];
    data.par_chunks_mut(nb)
        .enumerate()
        .with_min_len(CELL_MIN_LEN)
        .for_each_init(
            || vec![0.0f64; nb],
            |acc, (c, out)| {
                let (idx, val) = counts.row(c);
                plan.cell(idx, val, params.cap, acc);
                out.iter_mut()
                    .zip(acc.iter())
                    .for_each(|(o, &v)| *o = narrow(v));
            },
        );
    Ok(SmoothedBins {
        data,
        n_cells,
        bins: plan.bins,
        kept: plan.kept,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::filter_genes;
    use crate::reference::tests::{MHC, lambdas_col, load_sim, read_tsv};

    /// Dense naive path, the oracle for the fused one: full E per cell,
    /// centre, per-chromosome running mean by direct window sums, bin means.
    fn dense_naive(
        counts: &CsrCounts,
        order: &GenomeOrder,
        keep: &[bool],
        lambda_bar: &[f64],
        p: &SmoothParams,
    ) -> Vec<f64> {
        let kept: Vec<usize> = (0..order.n_genes()).filter(|&i| keep[i]).collect();
        let nk = kept.len();
        let mut runs: Vec<(u32, usize, usize)> = Vec::new();
        for (k, &i) in kept.iter().enumerate() {
            match runs.last_mut() {
                Some(r) if r.0 == order.chrom[i] => r.2 = k + 1,
                _ => runs.push((order.chrom[i], k, k + 1)),
            }
        }
        let bins = bin_layout(&runs, p.bin_size);
        let mut out = Vec::new();
        for c in 0..counts.n_cells() {
            let (idx, val) = counts.row(c);
            let mut y = vec![0.0f64; order.n_genes()];
            for (&j, &v) in idx.iter().zip(val) {
                y[order.rank[j as usize] as usize] = v as f64;
            }
            let lib: f64 = kept.iter().map(|&i| y[i]).sum();
            let mut e: Vec<f64> = kept
                .iter()
                .map(|&i| {
                    let x = if lib > 0.0 { y[i] / lib } else { 0.0 };
                    ((x * CPM + 1.0).log2() - (lambda_bar[i] * CPM + 1.0).log2())
                        .clamp(-p.cap, p.cap)
                })
                .collect();
            let m = e.iter().sum::<f64>() / nk as f64;
            e.iter_mut().for_each(|v| *v -= m);
            let mut s = vec![0.0; nk];
            for &(_, lo, hi) in &runs {
                for i in 0..hi - lo {
                    let (a, b) = window_at(i, hi - lo, p.window);
                    s[lo + i] = e[lo + a..lo + b].iter().sum::<f64>() / (b - a) as f64;
                }
            }
            for bin in &bins {
                let (a, b) = (bin.lo as usize, bin.hi as usize);
                out.push(s[a..b].iter().sum::<f64>() / (b - a) as f64);
            }
        }
        out
    }

    /// R bins of one case: `(layout, cells, values cells x bins)`.
    fn r_case(case: &str) -> (Vec<Bin>, Vec<usize>, Vec<f64>) {
        let layout: Vec<Bin> = read_tsv("smooth_layout.tsv")
            .iter()
            .filter(|r| r[0] == case)
            .map(|r| Bin {
                chrom: r[2].parse().unwrap(),
                lo: r[3].parse().unwrap(),
                hi: r[4].parse().unwrap(),
            })
            .collect();
        let nb = layout.len();
        let rows: Vec<(usize, usize, f64)> = read_tsv("smooth_bins.tsv")
            .iter()
            .filter(|r| r[0] == case)
            .map(|r| {
                (
                    r[1].parse().unwrap(),
                    r[2].parse().unwrap(),
                    r[3].parse().unwrap(),
                )
            })
            .collect();
        let mut cells: Vec<usize> = rows.iter().map(|r| r.0).collect();
        cells.sort_unstable();
        cells.dedup();
        let mut v = vec![f64::NAN; cells.len() * nb];
        for (c, b, x) in rows {
            v[cells.binary_search(&c).unwrap() * nb + b] = x;
        }
        (layout, cells, v)
    }

    /// Max absolute difference over the R cells.
    fn max_diff<T: CnvFloat>(ours: &SmoothedBins<T>, cells: &[usize], want: &[f64]) -> f64 {
        let nb = ours.bins.len();
        let mut d = 0.0f64;
        for (k, &c) in cells.iter().enumerate() {
            for b in 0..nb {
                let o = ours.data[c * nb + b].to_f64().unwrap();
                d = d.max((o - want[k * nb + b]).abs());
            }
        }
        d
    }

    #[test]
    fn test_bin_layout_merges_short_remainder() {
        // 250 -> 100,100,50 (50 is not below 50); 240 -> 100,140
        let b = bin_layout(&[(1, 0, 250), (2, 250, 490), (3, 490, 520)], 100);
        let spans: Vec<(u32, u32)> = b.iter().map(|x| (x.lo, x.hi)).collect();
        assert_eq!(
            spans,
            vec![
                (0, 100),
                (100, 200),
                (200, 250),
                (250, 350),
                (350, 490),
                (490, 520)
            ]
        );
    }

    #[test]
    fn test_smooth_bins_match_r_per_chromosome() {
        let sim = load_sim();
        let counts = sim.counts();
        let o = &sim.order;
        let bar = lambdas_col(o, 1);
        let keep = filter_genes(&counts, o, &bar, None, &o.allowed_mask(&[MHC])).unwrap();
        let r_kept: Vec<u32> = read_tsv("smooth_kept.tsv")
            .iter()
            .filter(|r| r[0] == "d1_b100")
            .map(|r| o.rank[r[1].parse::<usize>().unwrap()])
            .collect();
        for (case, bin_size) in [("d1_b100", 100), ("d1_b37", 37)] {
            let p = SmoothParams::new(101, 3.0, bin_size);
            let (layout, cells, want) = r_case(case);
            let ours = smooth_bins::<f64>(&counts, o, &keep, &bar, Some(p)).unwrap();
            assert_eq!(ours.kept, r_kept);
            assert_eq!(ours.bins, layout);
            let d64 = max_diff(&ours, &cells, &want);
            let ours32 = smooth_bins::<f32>(&counts, o, &keep, &bar, Some(p)).unwrap();
            let d32 = max_diff(&ours32, &cells, &want);
            println!("{case}: max |diff| f64 {d64:.3e}, f32 {d32:.3e}");
            assert!(d64 < 1e-5, "{case} f64: {d64}");
            assert!(d32 < 1e-6, "{case} f32: {d32}");
        }
    }

    #[test]
    fn test_smooth_bins_match_unmodified_numbat_single_chromosome() {
        let sim = load_sim();
        let o = &sim.order;
        // genes of chromosome 1, renumbered in caller order
        let genes = read_tsv("ref_genes.tsv");
        let sub: Vec<usize> = (0..o.n_genes()).filter(|&g| genes[g][1] == "1").collect();
        let mut new_of = vec![u32::MAX; o.n_genes()];
        for (k, &g) in sub.iter().enumerate() {
            new_of[g] = k as u32;
        }
        let col =
            |j: usize| -> Vec<u64> { sub.iter().map(|&g| genes[g][j].parse().unwrap()).collect() };
        let o1 = GenomeOrder::new(&vec![1; sub.len()], &col(2), &col(3)).unwrap();
        let mut indptr = vec![0u32];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        let full = sim.counts();
        for c in 0..full.n_cells() {
            let (idx, val) = full.row(c);
            for (&j, &y) in idx.iter().zip(val) {
                if new_of[j as usize] != u32::MAX {
                    indices.push(new_of[j as usize]);
                    data.push(y);
                }
            }
            indptr.push(indices.len() as u32);
        }
        let counts = CsrCounts::new(&indptr, &indices, &data, sub.len()).unwrap();
        let bar_all: Vec<f64> = read_tsv("ref_lambdas.tsv")
            .iter()
            .map(|r| r[1].parse().unwrap())
            .collect();
        let bar = o1
            .to_genomic(&sub.iter().map(|&g| bar_all[g]).collect::<Vec<_>>())
            .unwrap();
        let keep = filter_genes(&counts, &o1, &bar, None, &vec![true; sub.len()]).unwrap();
        let r_kept: Vec<u32> = read_tsv("smooth_kept.tsv")
            .iter()
            .filter(|r| r[0] == "numbat1")
            .map(|r| o1.rank[new_of[r[1].parse::<usize>().unwrap()] as usize])
            .collect();
        let (layout, cells, want) = r_case("numbat1");
        let ours = smooth_bins::<f64>(&counts, &o1, &keep, &bar, None).unwrap();
        assert_eq!(ours.kept, r_kept);
        assert_eq!(ours.bins, layout);
        let d = max_diff(&ours, &cells, &want);
        println!("numbat1: max |diff| f64 {d:.3e}");
        assert!(d < 1e-10, "{d}");
    }

    #[test]
    fn test_sparse_fused_equals_dense_naive() {
        let sim = load_sim();
        let counts = sim.counts();
        let o = &sim.order;
        let bar = lambdas_col(o, 1);
        let keep = filter_genes(&counts, o, &bar, None, &o.allowed_mask(&[MHC])).unwrap();
        // default, an even window, a window wider than chromosome 21, tiny bins
        for p in [
            SmoothParams::default(),
            SmoothParams::new(10, 3.0, 7),
            SmoothParams::new(301, 1.5, 250),
            SmoothParams::new(1, 3.0, 1),
        ] {
            let ours = smooth_bins::<f64>(&counts, o, &keep, &bar, Some(p)).unwrap();
            let want = dense_naive(&counts, o, &keep, &bar, &p);
            let d = ours
                .data
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(d < 1e-10, "{p:?}: {d}");
        }
    }

    #[test]
    fn test_smooth_bins_rejects_bad_input() {
        let o = GenomeOrder::new(&[1, 1], &[1, 2], &[1, 2]).unwrap();
        let counts = CsrCounts::new(&[0, 1], &[0], &[3], 2).unwrap();
        let bar = [0.5, 0.5];
        assert!(smooth_bins::<f32>(&counts, &o, &[false, false], &bar, None).is_err());
        assert!(smooth_bins::<f32>(&counts, &o, &[true], &bar, None).is_err());
        let p = Some(SmoothParams::new(0, 3.0, 100));
        assert!(smooth_bins::<f32>(&counts, &o, &[true, true], &bar, p).is_err());
    }
}
