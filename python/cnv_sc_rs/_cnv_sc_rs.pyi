from typing import Any

import numpy as np

__version__: str
__core_version__: str

class CnvError(Exception): ...

def run_cnv(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_genes: int,
    chrom: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    ref_cells: np.ndarray,
    params: dict[str, Any],
    *,
    n_threads: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Flat result. Always: `bins`, `bin_chrom`, `bin_start`, `bin_end`,
    `bin_n_genes`, `cells`, `leaf`, `seg_*`. When a segment is non-neutral:
    `post_segs`, `entropy`, `confident`, `log_odds`, `mu`, `sigma`,
    `clone_prob`, `clone`, `p_tumour`, `genotypes`. Per-cell rows beyond
    `bins` and `leaf` follow `cells`."""

def default_params() -> dict[str, int | float]: ...
