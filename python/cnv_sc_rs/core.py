"""`infer_cnv`: expression-only CNV inference on a scipy count matrix."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from beartype import beartype
from scipy import sparse

from . import _cnv_sc_rs
from ._validate import (
    SparseMatrix,
    csr_counts,
    gene_mask,
    reference_cells,
    reference_groups,
    reference_profiles,
)

###########
# Globals #
###########

logger = logging.getLogger("cnv_sc_rs")

Genome = Literal["hg38", "hg19", "none"]

LOG_ODDS_COLUMNS: tuple[str, ...] = ("logbf", "loh", "del", "amp")

##########
# Params #
##########


@dataclass(frozen=True)
class CnvParams:
    """Tuning parameters. Defaults are numbat's (`run_numbat`), plus the
    binning, which is ours.

    Args:
        window: Running-mean window of the smoothed signal, in genes.
        cap: Symmetric cap on the per-gene log-ratio before smoothing.
        bin_size: Kept genes per bin of the smoothed signal.
        init_k: Leaf clusters cut from the cell hierarchy.
        min_cells: A pseudobulk needs more than this many cells.
        t: HMM transition probability between copy number states.
        logphi_min: `log2` fold change of a single-copy event.
        min_genes: Segments up to this many genes are smoothed away.
        min_llr: Segments with a lower log-likelihood ratio count as neutral.
        min_overlap: Overlap fraction that links two segments in the consensus.
        max_entropy: Segments with a higher mean `p_cnv` entropy are not
            confident.
        p_multi: Probability threshold of the multi-state test.
    """

    window: int = 101
    cap: float = 3.0
    bin_size: int = 100
    init_k: int = 3
    min_cells: int = 50
    t: float = 1e-5
    logphi_min: float = 0.25
    min_genes: int = 10
    min_llr: float = 5.0
    min_overlap: float = 0.45
    max_entropy: float = 0.5
    p_multi: float = 1.0 - 1e-4


##########
# Result #
##########


@dataclass(frozen=True)
class CnvResult:
    """Output of `infer_cnv`.

    Per-cell arrays have one row per input cell. Cells the core dropped (zero
    counts, or no usable reference) have `used == False`, `NaN` in float
    arrays and `-1` in integer ones. Posterior and clone arrays are `NaN` for
    every cell when no consensus segment is non-neutral.

    Args:
        used: Cells that went through the pipeline, `(n_cells,)` bool.
        bins: Smoothed log-ratio signal, `(n_cells, n_bins)` float32.
        bin_table: One row per bin: `chrom`, `start`, `end` (bp), `n_genes`.
        leaf: Leaf cluster of the cell hierarchy per cell, `-1` for none.
        segments: One row per consensus segment. `state` is the HMM label,
            `state_post` the retest call (`neu`, `amp`, `del`, `loh`),
            `p_*` the retest posteriors, `post_col` the column in `p_cnv`
            (null for neutral segments), `confident` whether it passed the
            entropy and LLR filters.
        p_cnv: Per-cell posterior probability of each non-neutral segment,
            `(n_cells, n_post)`.
        log_odds: Per-cell log-odds against neutral, `(n_cells, n_post, 4)`;
            the last axis is `LOG_ODDS_COLUMNS` (pooled log Bayes factor,
            LOH, deletion, gain).
        mu: Per-cell baseline `mu` of the Poisson-lognormal fit.
        sigma: Per-cell baseline `sigma`.
        p_tumour: `1 - p(normal genotype)` per cell.
        clone: Most probable genotype per cell, `0` is normal.
        clone_prob: Genotype posterior, `(n_cells, n_genotypes)`.
        genotypes: Per genotype, its events as `(segment row, state)`;
            `state` is `None` unless the segment is multi-state. Genotype 0
            is the empty, normal one.
        reference: Reference profile each cell correlates best with, an
            index into `reference_names`, `-1` for dropped cells.
        reference_names: Name per reference profile: the label with
            `reference_labels`, otherwise its column index (`"0"` for pooled
            reference cells).
        params: Parameters the run used.
        genome: Exclusion intervals the run used.
    """

    used: np.ndarray
    bins: np.ndarray
    bin_table: pl.DataFrame
    leaf: np.ndarray
    segments: pl.DataFrame
    p_cnv: np.ndarray
    log_odds: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    p_tumour: np.ndarray
    clone: np.ndarray
    clone_prob: np.ndarray
    genotypes: list[list[tuple[int, str | None]]]
    reference: np.ndarray
    reference_names: list[str]
    params: CnvParams
    genome: str


###########
# Helpers #
###########


@beartype
def _scatter(
    rows: np.ndarray, x: np.ndarray, n_cells: int, fill: float | int
) -> np.ndarray:
    """Place per-used-cell rows into a full-length array filled with `fill`."""
    out = np.full((n_cells, *x.shape[1:]), fill, dtype=x.dtype)
    out[rows] = x
    return out


@beartype
def _segments(raw: dict) -> pl.DataFrame:
    """Consensus segment table from the flat core output."""
    n_seg = len(raw["seg_chrom"])
    post_col = np.full(n_seg, -1, dtype=np.int64)
    entropy = np.full(n_seg, np.nan)
    confident = np.zeros(n_seg, dtype=bool)
    if "post_segs" in raw:
        cols = raw["post_segs"].astype(np.int64)
        post_col[cols] = np.arange(len(cols))
        entropy[cols] = raw["entropy"]
        confident[cols] = raw["confident"]
    return pl.DataFrame(
        {
            "chrom": raw["seg_chrom"],
            "seg": raw["seg_seg"],
            "start": raw["seg_start"],
            "end": raw["seg_end"],
            "n_genes": raw["seg_n_genes"],
            "bulk": raw["seg_bulk"],
            "state": raw["seg_state"],
            "state_post": raw["seg_state_post"],
            "phi_hat": raw["seg_phi_hat"],
            "s_phi": raw["seg_s_phi"],
            "llr": raw["seg_llr"],
            "p_neu": raw["seg_p_neu"],
            "p_amp": raw["seg_p_amp"],
            "p_del": raw["seg_p_del"],
            "p_loh": raw["seg_p_loh"],
            "multi_state": pl.Series(raw["seg_multi_state"], dtype=pl.List(pl.String)),
            "post_col": post_col,
            "entropy": entropy,
            "confident": confident,
        }
    ).with_columns(
        pl.col("bulk").replace(-1, None),
        pl.col("post_col").replace(-1, None),
    )


################
# Entry point  #
################


@beartype
def infer_cnv(
    X: SparseMatrix,
    chrom: npt.ArrayLike,
    start: npt.ArrayLike,
    end: npt.ArrayLike,
    reference: npt.ArrayLike,
    *,
    reference_labels: npt.ArrayLike | None = None,
    params: CnvParams | None = None,
    genome: Genome = "hg38",
    n_threads: int | None = None,
    verbose: bool = False,
) -> CnvResult:
    """Copy number inference from single cell expression.

    The expression path of numbat: smoothed log-ratio signal against the
    reference cells, a cell hierarchy, Poisson-lognormal HMMs on
    pseudobulks, consensus segments, then per-cell segment posteriors and a
    tumour/normal call per cell.

    Genes off chromosomes 1 to 22, or with missing coordinates, are dropped.

    Args:
        X: Cells x genes raw UMI counts, any scipy sparse format. Not
            normalised, not log-transformed.
        chrom: Chromosome per gene (`1`, `"1"` or `"chr1"`).
        start: Gene start in bp per gene.
        end: Gene end in bp per gene.
        reference: Normal reference cells as a boolean mask over rows of `X`
            or integer row indices; or precomputed profiles, a float
            `(n_genes, R)` array of proportions in the gene order of `X`.
            Each cell is compared against the profile it correlates best
            with. Profiles are subset to the kept genes, not renormalised.
        reference_labels: Label per cell (length `n_cells`), read on the
            reference cells only. Builds one profile per label instead of
            pooling all reference cells into one. Not with profiles.
        params: Tuning parameters, `None` for the defaults.
        genome: Build of the default exclusion intervals (MHC and the IG/TR
            loci), or `"none"` to exclude nothing.
        n_threads: Threads for this call, `None` for all cores.
        verbose: Print stage timings to stdout.

    Returns:
        The result; see `CnvResult`.

    Raises:
        ValueError: For malformed inputs.
        CnvError: When the core fails, for instance with too few cells per
            pseudobulk.
    """
    params = params or CnvParams()
    csr = sparse.csr_matrix(X)
    n_cells, n_genes = csr.shape
    keep, chrom_ids, s, e = gene_mask(chrom, start, end, n_genes)
    ref_arr = np.asarray(reference)
    if ref_arr.ndim == 2 and ref_arr.dtype.kind == "f":
        if reference_labels is not None:
            raise ValueError("reference_labels needs reference cells, not profiles")
        refs = {"ref_profiles": reference_profiles(ref_arr, keep)}
        names = [str(r) for r in range(ref_arr.shape[1])]
    else:
        ref = reference_cells(ref_arr, n_cells)
        refs = {"ref_cells": ref}
        names = ["0"]
        if reference_labels is not None:
            refs["ref_group"], names = reference_groups(reference_labels, ref, n_cells)
    if not keep.all():
        csr = csr[:, np.flatnonzero(keep)]
    indptr, indices, data = csr_counts(csr)

    raw = _cnv_sc_rs.run_cnv(
        indptr,
        indices,
        data,
        int(keep.sum()),
        chrom_ids,
        s,
        e,
        {**dataclasses.asdict(params), "exclude": genome},
        **refs,
        n_threads=n_threads,
        verbose=verbose,
    )

    rows = raw["cells"].astype(np.int64)
    used = np.zeros(n_cells, dtype=bool)
    used[rows] = True
    if "log_odds" in raw:
        log_odds = _scatter(rows, raw["log_odds"], n_cells, np.nan)
        mu = _scatter(rows, raw["mu"], n_cells, np.nan)
        sigma = _scatter(rows, raw["sigma"], n_cells, np.nan)
    else:
        logger.warning("no non-neutral consensus segment; no per-cell posteriors")
        log_odds = np.full((n_cells, 0, len(LOG_ODDS_COLUMNS)), np.nan, np.float32)
        mu = np.full(n_cells, np.nan, np.float32)
        sigma = np.full(n_cells, np.nan, np.float32)
    if "clone" in raw:
        p_tumour = _scatter(rows, raw["p_tumour"], n_cells, np.nan)
        clone = _scatter(rows, raw["clone"].astype(np.int32), n_cells, -1)
        clone_prob = _scatter(rows, raw["clone_prob"], n_cells, np.nan)
        post_segs = raw["post_segs"]
        genotypes = [
            [(int(post_segs[col]), state) for col, state in g] for g in raw["genotypes"]
        ]
    else:
        p_tumour = np.full(n_cells, np.nan, np.float32)
        clone = np.full(n_cells, -1, np.int32)
        clone_prob = np.full((n_cells, 0), np.nan, np.float32)
        genotypes = []

    with np.errstate(over="ignore"):
        p_cnv = 1.0 / (1.0 + np.exp(-log_odds[..., 0]))

    return CnvResult(
        used=used,
        bins=raw["bins"],
        bin_table=pl.DataFrame(
            {
                "chrom": raw["bin_chrom"],
                "start": raw["bin_start"],
                "end": raw["bin_end"],
                "n_genes": raw["bin_n_genes"],
            }
        ),
        leaf=raw["leaf"],
        segments=_segments(raw),
        p_cnv=p_cnv,
        log_odds=log_odds,
        mu=mu,
        sigma=sigma,
        p_tumour=p_tumour,
        clone=clone,
        clone_prob=clone_prob,
        genotypes=genotypes,
        reference=raw["refs"],
        reference_names=names,
        params=params,
        genome=genome,
    )
