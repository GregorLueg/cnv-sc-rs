"""Scanpy-style `infer_cnv_adata` over an `AnnData`.

Reads raw counts from `adata.X` or a layer and gene loci from `adata.var`,
then writes the result into `obs`, `obsm` and `uns` under `key_added`. Every
slot is h5ad-writable. Scanpy itself is never imported.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
from anndata import AnnData
from beartype import beartype
from scipy import sparse

from .core import CnvParams, CnvResult, Genome, infer_cnv


@beartype
def _uns(res: CnvResult) -> dict:
    """Bin table, segments and run settings as plain numpy columns."""
    bins = res.bin_table
    chrom = bins["chrom"].to_numpy()
    first = np.flatnonzero(np.r_[True, chrom[1:] != chrom[:-1]])
    segs = res.segments.with_columns(
        res.segments["multi_state"].list.join(",").fill_null(""),
        res.segments["bulk"].fill_null(-1),
        res.segments["post_col"].fill_null(-1),
    )
    return {
        "chr_pos": {f"chr{chrom[i]}": int(i) for i in first},
        "bins": {c: bins[c].to_numpy() for c in bins.columns},
        "segments": {c: segs[c].to_numpy() for c in segs.columns},
        "reference_names": np.array(res.reference_names, dtype=object),
        "params": dataclasses.asdict(res.params),
        "genome": res.genome,
    }


@beartype
def infer_cnv_adata(
    adata: AnnData,
    *,
    reference_key: str,
    reference_cat: str | Sequence[str],
    reference_group_key: str | None = None,
    layer: str | None = None,
    chrom_key: str = "chromosome",
    start_key: str = "start",
    end_key: str = "end",
    key_added: str = "cnv",
    params: CnvParams | None = None,
    genome: Genome = "hg38",
    n_threads: int | None = None,
    verbose: bool = False,
) -> CnvResult:
    """Run `infer_cnv` on an `AnnData` and store the result in it.

    Writes, with `key_added="cnv"`:

    - `obsm["X_cnv"]`: smoothed signal, cells x bins.
    - `obsm["cnv_p_cnv"]`: per-cell posterior of each non-neutral segment.
    - `obs["cnv_p_tumour"]`, `obs["cnv_clone"]`, `obs["cnv_leaf"]`,
      `obs["cnv_reference"]` (index into `reference_names`).
    - `uns["cnv"]`: `chr_pos` (first bin per chromosome), `bins`,
      `segments` (columns of `CnvResult.segments`; `multi_state` joined with
      commas, nulls as `-1`), `reference_names`, `params`, `genome`.

    Args:
        adata: Cells x genes. Must hold raw counts in `layer` (or `X`).
        reference_key: `obs` column that labels the normal cells.
        reference_cat: Value(s) of `reference_key` that are normal.
        reference_group_key: `obs` column that splits the normal cells into
            one profile per value, e.g. the cell type. Can be
            `reference_key` itself. `None` pools them into one profile.
        layer: Layer with raw counts, `None` for `adata.X`.
        chrom_key: `var` column with the chromosome per gene.
        start_key: `var` column with the gene start.
        end_key: `var` column with the gene end.
        key_added: Prefix of the written slots.
        params: Tuning parameters, `None` for the defaults.
        genome: Build of the default exclusion intervals, or `"none"`.
        n_threads: Threads for this call, `None` for all cores.
        verbose: Print stage timings to stdout.

    Returns:
        The full `CnvResult`, which also carries `log_odds`, `clone_prob` and
        `genotypes`.

    Raises:
        KeyError: For a missing `obs`/`var` column or layer.
        ValueError: For a reference that matches no cell, or a normal cell
            without a `reference_group_key` value.
    """
    X = sparse.csr_matrix(adata.X if layer is None else adata.layers[layer])
    for key in (chrom_key, start_key, end_key):
        if key not in adata.var:
            raise KeyError(f"adata.var has no column {key!r}; see the scanpy guide")
    for key in (reference_key, reference_group_key):
        if key is not None and key not in adata.obs:
            raise KeyError(f"adata.obs has no column {key!r}")
    cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
    ref = adata.obs[reference_key].isin(cats).to_numpy()
    if not ref.any():
        raise ValueError(f"no cell has {reference_key} in {cats}")

    res = infer_cnv(
        X,
        adata.var[chrom_key].to_numpy(),
        adata.var[start_key].to_numpy(),
        adata.var[end_key].to_numpy(),
        ref,
        reference_labels=None
        if reference_group_key is None
        else adata.obs[reference_group_key].to_numpy(),
        params=params,
        genome=genome,
        n_threads=n_threads,
        verbose=verbose,
    )
    adata.obsm[f"X_{key_added}"] = res.bins
    adata.obsm[f"{key_added}_p_cnv"] = res.p_cnv
    adata.obs[f"{key_added}_p_tumour"] = res.p_tumour
    adata.obs[f"{key_added}_clone"] = res.clone
    adata.obs[f"{key_added}_leaf"] = res.leaf
    adata.obs[f"{key_added}_reference"] = res.reference
    adata.uns[key_added] = _uns(res)
    return res
