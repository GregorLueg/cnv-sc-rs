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
            np.zeros(1, np.uint32),
            {"windw": 3},
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
