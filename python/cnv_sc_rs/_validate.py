"""Input coercion: scipy counts, gene loci and reference cells to what the core
reads (`u32` CSR, `u32` chromosome ids, `u64` positions).

The core has no notion of autosomes or genome build. This layer keeps genes on
chromosomes 1 to 22 with finite coordinates, as numbat does, and drops the
rest by column subsetting.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import numpy.typing as npt
from beartype import beartype
from scipy import sparse

###########
# Globals #
###########

logger = logging.getLogger("cnv_sc_rs")

U32_MAX: int = np.iinfo(np.uint32).max

AUTOSOME: re.Pattern[str] = re.compile(
    r"^(?:chr)?([1-9]|1[0-9]|2[0-2])$", re.IGNORECASE
)

SparseMatrix = sparse.spmatrix | sparse.sparray

#########
# Genes #
#########


@beartype
def chromosome_ids(chrom: npt.ArrayLike) -> np.ndarray:
    """Autosome number per gene, 0 for anything else.

    Accepts `1`, `"1"`, `"chr1"`, `"CHR1"`. X, Y, MT, scaffolds and missing
    values map to 0.

    Args:
        chrom: Chromosome label per gene.

    Returns:
        `uint32` array, 1 to 22 for autosomes and 0 otherwise.
    """
    out = np.zeros(len(np.asarray(chrom, dtype=object)), dtype=np.uint32)
    for i, c in enumerate(np.asarray(chrom, dtype=object)):
        m = AUTOSOME.match(str(c).strip())
        if m is not None:
            out[i] = int(m.group(1))
    return out


@beartype
def gene_mask(
    chrom: npt.ArrayLike, start: npt.ArrayLike, end: npt.ArrayLike, n_genes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genes the core can use, with their loci.

    Args:
        chrom: Chromosome label per gene.
        start: Gene start in bp per gene.
        end: Gene end in bp per gene.
        n_genes: Number of columns of the count matrix.

    Returns:
        Boolean keep mask over all genes, then chromosome id (`uint32`), start
        and end (`uint64`) of the kept genes.

    Raises:
        ValueError: On a length mismatch, negative or reversed coordinates, or
            when no gene is left.
    """
    ids = chromosome_ids(chrom)
    s = np.asarray(start, dtype=np.float64)
    e = np.asarray(end, dtype=np.float64)
    for name, arr in (("chrom", ids), ("start", s), ("end", e)):
        if arr.shape != (n_genes,):
            raise ValueError(f"{name} has shape {arr.shape}, expected ({n_genes},)")
    keep = (ids > 0) & np.isfinite(s) & np.isfinite(e)
    if np.any((s[keep] < 0) | (e[keep] < s[keep])):
        raise ValueError("gene coordinates must satisfy 0 <= start <= end")
    if not keep.any():
        raise ValueError(
            "no gene on chromosomes 1-22 with finite start/end; "
            "check the chromosome labels and coordinates"
        )
    if (dropped := int(n_genes - keep.sum())) > 0:
        logger.info("dropping %d genes off chromosomes 1-22 or without loci", dropped)
    return keep, ids[keep], s[keep].astype(np.uint64), e[keep].astype(np.uint64)


##########
# Counts #
##########


@beartype
def csr_counts(X: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR arrays of a raw count matrix as `uint32`.

    Sums duplicate entries, on a copy. Float data is accepted when every
    value is a whole number.

    Args:
        X: Cells x genes raw counts.

    Returns:
        `indptr`, `indices`, `data`, all `uint32` and contiguous.

    Raises:
        ValueError: For negative, non-integer or too large counts, or more
            nonzeros than `uint32` can index.
    """
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
    if X.nnz > U32_MAX:
        raise ValueError(f"{X.nnz} nonzeros exceed the uint32 limit of the core")
    data = X.data
    if data.dtype.kind not in "uif":
        raise ValueError(f"unsupported count dtype {data.dtype}")
    if data.size:
        if data.min() < 0:
            raise ValueError("counts must be non-negative; pass raw counts")
        if data.max() > U32_MAX:
            raise ValueError("counts exceed uint32")
        if data.dtype.kind == "f" and np.any(data != np.floor(data)):
            raise ValueError(
                "counts are not whole numbers; pass raw counts, not normalised data"
            )
    return (
        np.ascontiguousarray(X.indptr, dtype=np.uint32),
        np.ascontiguousarray(X.indices, dtype=np.uint32),
        np.ascontiguousarray(data, dtype=np.uint32),
    )


#############
# Reference #
#############


@beartype
def reference_cells(reference: npt.ArrayLike, n_cells: int) -> np.ndarray:
    """Reference cells as sorted unique row indices.

    Args:
        reference: Boolean mask of length `n_cells` or integer row indices.
        n_cells: Number of rows of the count matrix.

    Returns:
        `uint32` row indices.

    Raises:
        ValueError: For an empty reference, a mask of the wrong length or an
            index out of range.
    """
    r = np.asarray(reference)
    if r.dtype == bool:
        if r.shape != (n_cells,):
            raise ValueError(
                f"reference mask has shape {r.shape}, expected ({n_cells},)"
            )
        r = np.flatnonzero(r)
    elif r.dtype.kind not in "ui":
        raise ValueError("reference must be a boolean mask or integer row indices")
    r = np.unique(r.ravel())
    if r.size == 0:
        raise ValueError("reference is empty")
    if r[0] < 0 or r[-1] >= n_cells:
        raise ValueError(f"reference indices must lie in [0, {n_cells})")
    return r.astype(np.uint32)
