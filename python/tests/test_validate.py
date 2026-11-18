from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from cnv_sc_rs._validate import chromosome_ids, csr_counts, gene_mask, reference_cells


def test_chromosome_ids() -> None:
    got = chromosome_ids(
        ["1", "chr2", "CHR22", 7, "X", "chrY", "MT", "23", None, "GL000"]
    )
    np.testing.assert_array_equal(got, [1, 2, 22, 7, 0, 0, 0, 0, 0, 0])


def test_gene_mask_drops_missing_loci() -> None:
    keep, ids, s, e = gene_mask(["1", "1", "X"], [10, np.nan, 5], [20, 30, 6], 3)
    np.testing.assert_array_equal(keep, [True, False, False])
    assert (ids.dtype, s.dtype, e.dtype) == (np.uint32, np.uint64, np.uint64)


def test_gene_mask_rejects_reversed() -> None:
    with pytest.raises(ValueError, match="start <= end"):
        gene_mask(["1"], [20], [10], 1)


def test_gene_mask_rejects_length() -> None:
    with pytest.raises(ValueError, match="shape"):
        gene_mask(["1", "2"], [1], [2], 2)


def test_gene_mask_rejects_no_autosome() -> None:
    with pytest.raises(ValueError, match="no gene"):
        gene_mask(["X"], [1], [2], 1)


def test_counts_accepts_integral_float() -> None:
    X = sparse.csr_matrix(np.array([[0, 2.0], [3.0, 0]], dtype=np.float32))
    _, _, data = csr_counts(X)
    assert data.dtype == np.uint32
    np.testing.assert_array_equal(data, [2, 3])


def test_counts_rejects_normalised() -> None:
    with pytest.raises(ValueError, match="whole numbers"):
        csr_counts(sparse.csr_matrix(np.array([[0.5, 1.0]])))


def test_counts_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        csr_counts(sparse.csr_matrix(np.array([[-1, 1]])))


def test_counts_sums_duplicates() -> None:
    X = sparse.csr_matrix(([1, 2], [1, 1], [0, 2]), shape=(1, 2))
    _, indices, data = csr_counts(X)
    np.testing.assert_array_equal(indices, [1])
    np.testing.assert_array_equal(data, [3])


def test_reference_mask_and_indices() -> None:
    np.testing.assert_array_equal(
        reference_cells(np.array([True, False, True]), 3), [0, 2]
    )
    np.testing.assert_array_equal(reference_cells([2, 0, 2], 3), [0, 2])


@pytest.mark.parametrize("ref", [[], [3], [-1], np.array([True, False])])
def test_reference_rejects(ref: object) -> None:
    with pytest.raises(ValueError):
        reference_cells(ref, 3)
