from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cnv_sc_rs as cnv

from .conftest import N_REF, Sim

anndata = pytest.importorskip("anndata")


@pytest.fixture
def adata(sim: Sim) -> anndata.AnnData:
    n = sim.X.shape[0]
    ad = anndata.AnnData(
        X=sim.X.astype(np.float32),
        obs={"cell_type": np.where(np.arange(n) < N_REF, "T cell", "epithelial")},
        var={"chromosome": sim.chrom, "start": sim.start, "end": sim.end},
    )
    ad.layers["counts"] = sim.X
    return ad


def test_writes_slots_and_round_trips(adata: anndata.AnnData, tmp_path: Path) -> None:
    res = cnv.infer_cnv_adata(
        adata,
        reference_key="cell_type",
        reference_cat="T cell",
        layer="counts",
        genome="none",
    )
    assert adata.obsm["X_cnv"].shape == res.bins.shape
    assert adata.obsm["cnv_p_cnv"].shape == res.p_cnv.shape
    np.testing.assert_array_equal(adata.obs["cnv_p_tumour"].to_numpy(), res.p_tumour)
    assert adata.uns["cnv"]["chr_pos"]["chr1"] == 0
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    assert set(back.uns["cnv"]["segments"]) == set(res.segments.columns)


def test_missing_var_column(adata: anndata.AnnData) -> None:
    del adata.var["start"]
    with pytest.raises(KeyError, match="start"):
        cnv.infer_cnv_adata(adata, reference_key="cell_type", reference_cat="T cell")


def test_reference_matches_nothing(adata: anndata.AnnData) -> None:
    with pytest.raises(ValueError, match="no cell"):
        cnv.infer_cnv_adata(adata, reference_key="cell_type", reference_cat="B cell")


def test_reference_group_key(adata: anndata.AnnData, tmp_path: Path) -> None:
    n = adata.n_obs
    adata.obs["subtype"] = np.where(np.arange(n) % 2 == 0, "CD4", "CD8")
    res = cnv.infer_cnv_adata(
        adata,
        reference_key="cell_type",
        reference_cat="T cell",
        reference_group_key="subtype",
        layer="counts",
        genome="none",
    )
    assert res.reference_names == ["CD4", "CD8"]
    np.testing.assert_array_equal(adata.obs["cnv_reference"].to_numpy(), res.reference)
    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    assert list(back.uns["cnv"]["reference_names"]) == ["CD4", "CD8"]


def test_reference_group_key_missing_on_normal_cell(adata: anndata.AnnData) -> None:
    adata.obs["subtype"] = None
    with pytest.raises(ValueError, match="missing"):
        cnv.infer_cnv_adata(
            adata,
            reference_key="cell_type",
            reference_cat="T cell",
            reference_group_key="subtype",
        )
