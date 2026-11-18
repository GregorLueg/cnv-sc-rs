# Quickstart

```python
import numpy as np
import cnv_sc_rs as cnv

# X       scipy.sparse, cells x genes, raw counts
# chrom   per gene: 1, "1" or "chr1"
# start   per gene, bp
# end     per gene, bp
# normal  boolean mask over cells, or integer row indices

res = cnv.infer_cnv(X, chrom, start, end, reference=normal, genome="hg38")
```

`genome` picks the exclusion intervals: MHC and the IG/TR loci on `"hg38"`
or `"hg19"`, or `"none"`. Match it to the build of your gene coordinates.

## Reading the result

`res` is a `CnvResult`. Per-cell arrays have one row per input cell; cells
the pipeline dropped (zero counts) have `res.used == False`, `NaN` in float
arrays and `-1` in integer ones.

```python
# consensus segments, a polars DataFrame
calls = res.segments.filter(res.segments["confident"])
calls.select("chrom", "start", "end", "n_genes", "state_post", "llr")

res.p_tumour  # per cell, 1 - p(normal genotype)
res.clone  # most probable genotype per cell, 0 is normal
res.genotypes  # per genotype: [(segment row, state), ...]
res.p_cnv  # cells x non-neutral segments, posterior of each CNV
res.bins  # cells x bins, smoothed log-ratio signal
res.bin_table  # chrom, start, end, n_genes per bin
```

`segments["post_col"]` maps a segment row to its column in `p_cnv` and
`log_odds`. `state` is the HMM label (`amp1`, `del1`, `loh1`, ...),
`state_post` the retest call (`neu`, `amp`, `del`, `loh`).

If no segment comes out non-neutral, the posteriors are empty and
`p_tumour` is `NaN` throughout, with a warning on the `cnv_sc_rs` logger.

## Parameters

Defaults are numbat's. Change them through `CnvParams`:

```python
params = cnv.CnvParams(min_cells=30, init_k=4, min_llr=10.0)
res = cnv.infer_cnv(X, chrom, start, end, normal, params=params)
```

The knobs people turn first:

| Parameter     | Default | Effect                                              |
|---------------|---------|-----------------------------------------------------|
| `min_cells`   | 50      | Smallest pseudobulk; lower it for small samples     |
| `init_k`      | 3       | Leaf clusters cut from the cell hierarchy           |
| `min_llr`     | 5.0     | Evidence a segment needs to count as a CNV          |
| `max_entropy` | 0.5     | Per-cell uncertainty a segment may have and stay confident |

## Threads and determinism

`n_threads=` caps the threads for one call; the default uses every core.
The result does not depend on the thread count.
