# cnv-sc-rs

Copy number inference from single cell expression, in Rust. A port of the
expression path of [numbat](https://github.com/kharchenkolab/numbat) and
hahmmr (both MIT): inferCNV-style smoothing, a Poisson-lognormal HMM on
pseudobulks, consensus segments and per-cell segment posteriors.

```bash
pip install cnv-sc-rs            # core
pip install "cnv-sc-rs[anndata]" # plus infer_cnv_adata
```

```python
import cnv_sc_rs as cnv

# X: cells x genes raw counts (scipy sparse), loci per gene, normal cells
res = cnv.infer_cnv(X, chrom, start, end, reference=is_normal)
res.segments  # polars DataFrame, one row per consensus segment
res.p_tumour  # per cell
```

With scanpy:

```python
cnv.infer_cnv_adata(
    adata, reference_key="cell_type", reference_cat=["T cell", "B cell"], layer="counts"
)
```

Docs: https://gregorlueg.github.io/cnv-sc-rs/
