"""Copy number inference from single cell expression, in Rust.

A port of the expression path of numbat (Gao et al., Nature Biotechnology
2023) and hahmmr. Raw counts, gene loci and a set of normal reference cells
in; smoothed signal, consensus CNV segments, per-cell segment posteriors and
a per-cell tumour probability out.

    >>> import cnv_sc_rs as cnv
    >>> res = cnv.infer_cnv(X, chrom, start, end, reference=is_normal)
    >>> res.segments.filter(res.segments["confident"])

With `anndata` installed, `infer_cnv_adata` reads from and writes into an
`AnnData` the scanpy way.
"""

from typing import TYPE_CHECKING

from ._cnv_sc_rs import CnvError, __core_version__, __version__
from .core import LOG_ODDS_COLUMNS, CnvParams, CnvResult, infer_cnv

if TYPE_CHECKING:
    from ._anndata import infer_cnv_adata

__all__ = [
    "LOG_ODDS_COLUMNS",
    "CnvError",
    "CnvParams",
    "CnvResult",
    "__core_version__",
    "__version__",
    "infer_cnv",
    "infer_cnv_adata",
]


def __getattr__(name: str) -> object:
    # Lazy, so `anndata` stays an optional dependency.
    if name == "infer_cnv_adata":
        from ._anndata import infer_cnv_adata

        return infer_cnv_adata
    raise AttributeError(f"module 'cnv_sc_rs' has no attribute {name!r}")
