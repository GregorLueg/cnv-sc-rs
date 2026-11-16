//! Gene coordinates, genomic order and chromosome ranges (SPEC 1), plus the
//! cells x genes CSR count view every stage reads.
//!
//! Genes arrive in the caller's order. [`GenomeOrder`] sorts them once by
//! (chromosome, start, end, caller index); all per-gene vectors downstream
//! are in that genomic order.

use rayon::prelude::*;

use crate::errors::CnvErrors;

///////////////
// CsrCounts //
///////////////

/// Cells x genes raw counts in CSR, genes in caller order (SPEC 1).
#[derive(Clone, Copy, Debug)]
pub struct CsrCounts<'a> {
    /// Row offsets, length `n_cells + 1`
    pub indptr: &'a [u32],
    /// Gene index per nonzero, caller order, `< n_genes`
    pub indices: &'a [u32],
    /// Count per nonzero
    pub data: &'a [u32],
    /// Number of genes (columns)
    pub n_genes: usize,
}

impl<'a> CsrCounts<'a> {
    /// Validate and wrap a CSR count matrix.
    ///
    /// ### Params
    ///
    /// * `indptr` - Row offsets, length `n_cells + 1`, non-decreasing, from 0
    /// * `indices` - Gene index per nonzero, caller order
    /// * `data` - Count per nonzero
    /// * `n_genes` - Number of genes
    ///
    /// ### Returns
    ///
    /// The view, or [`CnvErrors::InvalidCounts`] / [`CnvErrors::ShapeMismatch`].
    pub fn new(
        indptr: &'a [u32],
        indices: &'a [u32],
        data: &'a [u32],
        n_genes: usize,
    ) -> Result<Self, CnvErrors> {
        if indptr.is_empty() || indptr[0] != 0 {
            return Err(CnvErrors::InvalidCounts {
                what: "indptr must start at 0",
            });
        }
        if indptr.windows(2).any(|w| w[0] > w[1]) {
            return Err(CnvErrors::InvalidCounts {
                what: "indptr decreases",
            });
        }
        let nnz = *indptr.last().expect("non-empty") as usize;
        for (what, len) in [("indices", indices.len()), ("data", data.len())] {
            if len != nnz {
                return Err(CnvErrors::ShapeMismatch {
                    what,
                    expected: nnz,
                    got: len,
                });
            }
        }
        if indices.iter().any(|&j| j as usize >= n_genes) {
            return Err(CnvErrors::InvalidCounts {
                what: "gene index out of range",
            });
        }
        Ok(Self {
            indptr,
            indices,
            data,
            n_genes,
        })
    }

    /// Number of cells.
    ///
    /// ### Returns
    ///
    /// Rows of the matrix.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.indptr.len() - 1
    }

    /// Nonzeros of one cell.
    ///
    /// ### Params
    ///
    /// * `c` - Cell index
    ///
    /// ### Returns
    ///
    /// `(gene indices, counts)`, caller gene order.
    #[inline]
    pub fn row(&self, c: usize) -> (&'a [u32], &'a [u32]) {
        let (lo, hi) = (self.indptr[c] as usize, self.indptr[c + 1] as usize);
        (&self.indices[lo..hi], &self.data[lo..hi])
    }

    /// Per-gene count sums over a set of cells.
    ///
    /// Integer sums, so exact and independent of the thread count.
    ///
    /// ### Params
    ///
    /// * `cells` - Cells to sum over; `None` for all
    ///
    /// ### Returns
    ///
    /// Sum per gene, caller order, or an error for an out-of-range cell.
    pub fn gene_sums(&self, cells: Option<&[u32]>) -> Result<Vec<u64>, CnvErrors> {
        let n = self.n_genes;
        let add = |mut acc: Vec<u64>, c: usize| {
            let (idx, val) = self.row(c);
            for (&j, &y) in idx.iter().zip(val) {
                acc[j as usize] += y as u64;
            }
            acc
        };
        let merge = |mut a: Vec<u64>, b: Vec<u64>| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        };
        Ok(match cells {
            None => (0..self.n_cells())
                .into_par_iter()
                .fold(|| vec![0u64; n], add)
                .reduce(|| vec![0u64; n], merge),
            Some(cells) => {
                self.check_cells(cells)?;
                cells
                    .par_iter()
                    .fold(|| vec![0u64; n], |acc, &c| add(acc, c as usize))
                    .reduce(|| vec![0u64; n], merge)
            }
        })
    }

    /// Check that every cell index is in range.
    ///
    /// ### Params
    ///
    /// * `cells` - Cell indices
    ///
    /// ### Returns
    ///
    /// `Ok`, or [`CnvErrors::InvalidParameter`] naming the first bad index.
    pub(crate) fn check_cells(&self, cells: &[u32]) -> Result<(), CnvErrors> {
        match cells.iter().find(|&&c| c as usize >= self.n_cells()) {
            Some(&c) => Err(CnvErrors::InvalidParameter {
                name: "cells",
                value: c.to_string(),
                expected: "cell indices below n_cells",
            }),
            None => Ok(()),
        }
    }
}

/////////////////
// GenomeOrder //
/////////////////

/// Genomic sort of the caller's genes (SPEC 1).
#[derive(Clone, Debug)]
pub struct GenomeOrder {
    /// Caller index of the `i`-th gene in genomic order
    pub perm: Vec<u32>,
    /// Genomic index of caller gene `g`; inverse of `perm`
    pub rank: Vec<u32>,
    /// Chromosome per gene, genomic order
    pub chrom: Vec<u32>,
    /// Start in bp per gene, genomic order
    pub start: Vec<u64>,
    /// End in bp per gene, genomic order
    pub end: Vec<u64>,
    /// `(chrom, lo, hi)`: genes `[lo, hi)` of each chromosome, ascending ids
    pub chrom_ranges: Vec<(u32, usize, usize)>,
}

impl GenomeOrder {
    /// Sort genes by chromosome id, then start, end and caller index.
    ///
    /// ### Params
    ///
    /// * `chrom` - Chromosome id per gene, caller order
    /// * `start` - Start in bp, caller order
    /// * `end` - End in bp, caller order, `>= start`
    ///
    /// ### Returns
    ///
    /// The order, or an error on mismatched lengths, no genes, more than
    /// `u32::MAX` genes or `end < start`.
    pub fn new(chrom: &[u32], start: &[u64], end: &[u64]) -> Result<Self, CnvErrors> {
        let n = chrom.len();
        for (what, len) in [("start", start.len()), ("end", end.len())] {
            if len != n {
                return Err(CnvErrors::ShapeMismatch {
                    what,
                    expected: n,
                    got: len,
                });
            }
        }
        if n == 0 {
            return Err(CnvErrors::EmptyInput { what: "genes" });
        }
        if n > u32::MAX as usize {
            return Err(CnvErrors::InvalidParameter {
                name: "n_genes",
                value: n.to_string(),
                expected: "at most u32::MAX genes",
            });
        }
        if let Some(g) = (0..n).find(|&g| end[g] < start[g]) {
            return Err(CnvErrors::InvalidGeneInterval {
                gene: g,
                start: start[g],
                end: end[g],
            });
        }
        let mut perm: Vec<u32> = (0..n as u32).collect();
        perm.sort_unstable_by_key(|&g| {
            let g = g as usize;
            (chrom[g], start[g], end[g], g)
        });
        let mut rank = vec![0u32; n];
        for (i, &g) in perm.iter().enumerate() {
            rank[g as usize] = i as u32;
        }
        let chrom: Vec<u32> = perm.iter().map(|&g| chrom[g as usize]).collect();
        let start = perm.iter().map(|&g| start[g as usize]).collect();
        let end = perm.iter().map(|&g| end[g as usize]).collect();
        let chrom_ranges = chrom_runs(&chrom)?
            .into_iter()
            .map(|(lo, hi)| (chrom[lo], lo, hi))
            .collect();
        Ok(Self {
            perm,
            rank,
            chrom,
            start,
            end,
            chrom_ranges,
        })
    }

    /// Number of genes.
    ///
    /// ### Returns
    ///
    /// Gene count.
    #[inline]
    pub fn n_genes(&self) -> usize {
        self.perm.len()
    }

    /// Reorder a caller-order per-gene vector into genomic order.
    ///
    /// ### Params
    ///
    /// * `x` - One value per gene, caller order
    ///
    /// ### Returns
    ///
    /// The values in genomic order, or a shape error.
    pub fn to_genomic<T: Copy>(&self, x: &[T]) -> Result<Vec<T>, CnvErrors> {
        if x.len() != self.n_genes() {
            return Err(CnvErrors::ShapeMismatch {
                what: "per-gene vector",
                expected: self.n_genes(),
                got: x.len(),
            });
        }
        Ok(self.perm.iter().map(|&g| x[g as usize]).collect())
    }

    /// Genes outside every exclusion interval (SPEC 1).
    ///
    /// A gene overlaps `(chrom, lo, hi)` iff `start < hi` and `end > lo`,
    /// numbat's MHC test (N/R/utils.R:147-149) with `lo = 28510120` and
    /// `hi = 33480577` on chromosome 6.
    ///
    /// ### Params
    ///
    /// * `exclude` - `(chrom, lo, hi)` intervals in bp
    ///
    /// ### Returns
    ///
    /// Per gene in genomic order, `true` if no interval overlaps it.
    pub fn allowed_mask(&self, exclude: &[(u32, u64, u64)]) -> Vec<bool> {
        (0..self.n_genes())
            .map(|i| {
                !exclude.iter().any(|&(c, lo, hi)| {
                    self.chrom[i] == c && self.start[i] < hi && self.end[i] > lo
                })
            })
            .collect()
    }
}

/// Contiguous gene ranges per chromosome.
///
/// ### Params
///
/// * `chrom` - Chromosome per gene
///
/// ### Returns
///
/// `[lo, hi)` per run of equal ids, or an error if a chromosome reappears
/// after another one (not genomic order).
pub(crate) fn chrom_runs(chrom: &[u32]) -> Result<Vec<(usize, usize)>, CnvErrors> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0;
    for i in 1..=chrom.len() {
        if i == chrom.len() || chrom[i] != chrom[lo] {
            if runs.iter().any(|&(a, _)| chrom[a] == chrom[lo]) {
                return Err(CnvErrors::NotGenomicOrder {
                    what: "chromosome ids are not contiguous",
                });
            }
            runs.push((lo, i));
            lo = i;
        }
    }
    Ok(runs)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrom_runs_rejects_non_contiguous() {
        assert_eq!(chrom_runs(&[1, 1, 2, 2, 2]).unwrap(), vec![(0, 2), (2, 5)]);
        assert!(chrom_runs(&[1, 2, 1]).is_err());
        assert!(chrom_runs(&[]).unwrap().is_empty());
    }

    #[test]
    fn test_genome_order_sorts_and_inverts() {
        let o = GenomeOrder::new(&[2, 1, 2, 1], &[50, 30, 10, 5], &[60, 40, 20, 9]).unwrap();
        assert_eq!(o.perm, vec![3, 1, 2, 0]);
        for (i, &g) in o.perm.iter().enumerate() {
            assert_eq!(o.rank[g as usize] as usize, i);
        }
        assert_eq!(o.chrom_ranges, vec![(1, 0, 2), (2, 2, 4)]);
        assert_eq!(o.to_genomic(&[0, 1, 2, 3]).unwrap(), vec![3, 1, 2, 0]);
        assert!(GenomeOrder::new(&[1], &[5], &[4]).is_err());
    }

    #[test]
    fn test_allowed_mask_open_overlap() {
        let o =
            GenomeOrder::new(&[6, 6, 6, 1], &[10, 100, 200, 100], &[100, 150, 300, 150]).unwrap();
        // genomic order: (1,100) (6,10) (6,100) (6,200); interval (6, 100, 200)
        assert_eq!(
            o.allowed_mask(&[(6, 100, 200)]),
            vec![true, true, false, true]
        );
    }

    #[test]
    fn test_csr_counts_validates_and_sums() {
        assert!(CsrCounts::new(&[0, 2], &[0, 3], &[1, 1], 3).is_err());
        assert!(CsrCounts::new(&[0, 2, 1], &[0, 1], &[1, 1], 3).is_err());
        let m = CsrCounts::new(&[0, 2, 3], &[0, 2, 2], &[1, 4, 5], 3).unwrap();
        assert_eq!(m.gene_sums(None).unwrap(), vec![1, 0, 9]);
        assert_eq!(m.gene_sums(Some(&[1])).unwrap(), vec![0, 0, 5]);
        assert!(m.gene_sums(Some(&[2])).is_err());
    }
}
