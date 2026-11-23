from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import sparse

import cnv_sc_rs as cnv
from cnv_sc_rs import _cnv_sc_rs

from .conftest import N_REF, Sim


@pytest.fixture(scope="module")
def res(sim: Sim) -> cnv.CnvResult:
    return cnv.infer_cnv(
        sim.X, sim.chrom, sim.start, sim.end, sim.reference, genome="none"
    )


def test_defaults_match_crate() -> None:
    assert dataclasses.asdict(cnv.CnvParams()) == _cnv_sc_rs.default_params()


def test_calls_gain_and_loss(res: cnv.CnvResult) -> None:
    calls = res.segments.filter(res.segments["confident"])
    assert sorted(zip(calls["chrom"], calls["state_post"], strict=True)) == [
        (2, "amp"),
        (5, "del"),
    ]


def test_tumour_cells_separate(sim: Sim, res: cnv.CnvResult) -> None:
    query = np.arange(sim.truth.size) >= N_REF
    normal = res.p_tumour[query & (sim.truth == 0)]
    tumour = res.p_tumour[query & (sim.truth > 0)]
    assert np.mean(normal > 0.5) < 0.05
    assert np.mean(tumour > 0.5) > 0.9


def test_shapes(sim: Sim, res: cnv.CnvResult) -> None:
    n = sim.X.shape[0]
    n_post = res.segments["post_col"].drop_nulls().len()
    assert res.bins.shape == (n, res.bin_table.height)
    assert res.p_cnv.shape == (n, n_post)
    assert res.log_odds.shape == (n, n_post, len(cnv.LOG_ODDS_COLUMNS))
    assert res.clone_prob.shape == (n, len(res.genotypes))
    assert res.genotypes[0] == []


def test_dropped_cells_are_nan(sim: Sim) -> None:
    X = sparse.vstack([sim.X, sparse.csr_matrix((1, sim.X.shape[1]))]).tocsr()
    res = cnv.infer_cnv(X, sim.chrom, sim.start, sim.end, sim.reference, genome="none")
    assert not res.used[-1]
    assert np.isnan(res.p_tumour[-1])
    assert res.clone[-1] == -1


def test_non_autosomes_dropped(sim: Sim) -> None:
    chrom = sim.chrom.copy()
    chrom[:5] = "chrX"
    res = cnv.infer_cnv(sim.X, chrom, sim.start, sim.end, sim.reference, genome="none")
    assert res.bin_table["n_genes"].sum() <= sim.X.shape[1] - 5


def test_deterministic_across_threads(sim: Sim, res: cnv.CnvResult) -> None:
    one = cnv.infer_cnv(
        sim.X, sim.chrom, sim.start, sim.end, sim.reference, genome="none", n_threads=1
    )
    np.testing.assert_array_equal(one.bins, res.bins)
    np.testing.assert_array_equal(one.p_tumour, res.p_tumour)
    assert one.segments.equals(res.segments)


def test_unknown_param_rejected(sim: Sim) -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        _cnv_sc_rs.run_cnv(
            *(np.zeros(1, np.uint32),) * 3,
            1,
            np.ones(1, np.uint32),
            np.zeros(1, np.uint64),
            np.ones(1, np.uint64),
            {"windw": 3},
            ref_cells=np.zeros(1, np.uint32),
        )


def test_run_cnv_needs_one_reference_form() -> None:
    args = (
        *(np.zeros(1, np.uint32),) * 3,
        1,
        np.ones(1, np.uint32),
        np.zeros(1, np.uint64),
        np.ones(1, np.uint64),
        {},
    )
    with pytest.raises(ValueError, match="ref_cells"):
        _cnv_sc_rs.run_cnv(*args)
    with pytest.raises(ValueError, match="ref_cells"):
        _cnv_sc_rs.run_cnv(
            *args,
            ref_cells=np.zeros(1, np.uint32),
            ref_profiles=np.ones((1, 1), np.float32),
        )


def test_core_error_surfaces(sim: Sim) -> None:
    with pytest.raises(cnv.CnvError):
        cnv.infer_cnv(
            sim.X,
            sim.chrom,
            sim.start,
            sim.end,
            sim.reference,
            genome="none",
            params=cnv.CnvParams(min_cells=10_000),
        )


def test_pooled_reference_is_profile_zero(res: cnv.CnvResult) -> None:
    assert res.reference_names == ["0"]
    np.testing.assert_array_equal(res.reference, np.where(res.used, 0, -1))


@pytest.fixture(scope="module")
def labels(sim: Sim) -> np.ndarray:
    lab = np.full(sim.X.shape[0], None, dtype=object)
    lab[:N_REF] = np.where(np.arange(N_REF) % 2 == 0, "b", "a")
    return lab


@pytest.fixture(scope="module")
def res_labels(sim: Sim, labels: np.ndarray) -> cnv.CnvResult:
    return cnv.infer_cnv(
        sim.X,
        sim.chrom,
        sim.start,
        sim.end,
        sim.reference,
        reference_labels=labels,
        genome="none",
    )


def test_reference_labels_build_one_profile_each(res_labels: cnv.CnvResult) -> None:
    assert res_labels.reference_names == ["a", "b"]
    assert set(res_labels.reference[res_labels.used]) == {0, 1}
    calls = res_labels.segments.filter(res_labels.segments["confident"])
    assert sorted(calls["chrom"]) == [2, 5]


def test_reference_profiles_match_labels(
    sim: Sim, labels: np.ndarray, res_labels: cnv.CnvResult
) -> None:
    sums = np.stack(
        [np.asarray(sim.X[labels == k].sum(axis=0)).ravel() for k in ("a", "b")],
        axis=1,
    ).astype(np.float64)
    res = cnv.infer_cnv(
        sim.X, sim.chrom, sim.start, sim.end, sums / sums.sum(axis=0), genome="none"
    )
    assert res.reference_names == ["0", "1"]
    np.testing.assert_array_equal(res.reference, res_labels.reference)
    np.testing.assert_allclose(res.bins, res_labels.bins, atol=1e-4)


def test_reference_labels_rejected_with_profiles(sim: Sim) -> None:
    prof = np.ones((sim.X.shape[1], 1))
    with pytest.raises(ValueError, match="reference_labels"):
        cnv.infer_cnv(
            sim.X,
            sim.chrom,
            sim.start,
            sim.end,
            prof,
            reference_labels=np.zeros(sim.X.shape[0]),
        )
