# With scanpy

`infer_cnv_adata` reads raw counts and gene loci from an `AnnData`, runs
`infer_cnv` and writes the result back into it. Scanpy is not a dependency;
the snippets below just use it around the call.

## 1. Keep the raw counts

The method models UMI counts, so it needs them before any normalisation.
Stash them in a layer at the start of the analysis:

```python
import scanpy as sc

adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
adata.layers["counts"] = adata.X.copy()
# ... QC, normalisation, PCA, neighbours, UMAP, clustering, cell typing ...
```

## 2. Add gene loci to `adata.var`

`infer_cnv_adata` looks for `var["chromosome"]`, `var["start"]` and
`var["end"]` (change with `chrom_key=`, `start_key=`, `end_key=`). Genes
without loci are dropped, so a partial match is fine.

From Ensembl BioMart (GRCh38, needs network), keyed on the Ensembl IDs a
10x matrix carries in `var["gene_ids"]`:

```python
annot = (
    sc.queries.biomart_annotations(
        "hsapiens",
        ["ensembl_gene_id", "chromosome_name", "start_position", "end_position"],
    )
    .drop_duplicates("ensembl_gene_id")
    .set_index("ensembl_gene_id")
)

loci = annot.reindex(adata.var["gene_ids"])
adata.var["chromosome"] = loci["chromosome_name"].to_numpy()
adata.var["start"] = loci["start_position"].to_numpy()
adata.var["end"] = loci["end_position"].to_numpy()
```

Or from the GTF your count matrix was built with, which pins the build
exactly (10x references ship it as `genes/genes.gtf.gz`):

```python
import polars as pl

gtf = (
    pl.read_csv(
        "genes.gtf.gz",
        separator="\t",
        comment_prefix="#",
        has_header=False,
        quote_char=None,
        new_columns=[
            "chrom",
            "src",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attr",
        ],
    )
    .filter(pl.col("feature") == "gene")
    .select(
        pl.col("attr").str.extract(r'gene_id "([^".]+)').alias("gene_id"),
        "chrom",
        "start",
        "end",
    )
    .unique("gene_id", keep="first")
)
ids = pl.DataFrame({"gene_id": adata.var["gene_ids"].to_numpy()})
loci = ids.join(gtf, on="gene_id", how="left", maintain_order="left")
adata.var["chromosome"] = loci["chrom"].to_numpy()
adata.var["start"] = loci["start"].to_numpy()
adata.var["end"] = loci["end"].to_numpy()
```

Chromosome labels may be `1` or `chr1`; both work.

## 3. Pick the reference

The reference should be cells you trust to be diploid: immune, endothelial
or fibroblast cells from the same sample work well. Any `obs` column can
label them:

```python
import cnv_sc_rs as cnv

res = cnv.infer_cnv_adata(
    adata,
    reference_key="cell_type",
    reference_cat=["T cell", "B cell", "Macrophage"],
    layer="counts",
    genome="hg38",  # match the build of the loci
)
```

This writes:

| Slot                     | Content                                             |
|--------------------------|-----------------------------------------------------|
| `obsm["X_cnv"]`          | Smoothed log-ratio signal, cells x bins             |
| `obsm["cnv_p_cnv"]`      | Posterior of each non-neutral segment, per cell     |
| `obs["cnv_p_tumour"]`    | `1 - p(normal)` per cell                            |
| `obs["cnv_clone"]`       | Most probable genotype, `0` normal, `-1` not used   |
| `obs["cnv_leaf"]`        | Leaf cluster of the CNV hierarchy                   |
| `uns["cnv"]`             | `chr_pos`, `bins`, `segments`, `params`, `genome`   |

The returned `CnvResult` holds the same plus `log_odds`, `clone_prob`,
`genotypes` and the segment table as a polars DataFrame.

## 4. Look at it

Tumour probability and genotype on the expression UMAP:

```python
adata.obs["cnv_clone"] = adata.obs["cnv_clone"].astype("category")
sc.pl.umap(adata, color=["cell_type", "cnv_p_tumour", "cnv_clone"])
```

The called segments:

```python
res.segments.filter(res.segments["confident"]).select(
    "chrom", "start", "end", "n_genes", "state_post", "llr"
)
```

A genome-wide heatmap of the smoothed signal, cells ordered by genotype:

```python
import matplotlib.pyplot as plt
import numpy as np

order = np.argsort(res.clone, kind="stable")
chr_pos = adata.uns["cnv"]["chr_pos"]

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(
    adata.obsm["X_cnv"][order],
    aspect="auto",
    cmap="RdBu_r",
    vmin=-0.5,
    vmax=0.5,
    interpolation="nearest",
)
for pos in chr_pos.values():
    ax.axvline(pos - 0.5, color="black", lw=0.5)
ax.set_xticks(list(chr_pos.values()), [c.removeprefix("chr") for c in chr_pos])
ax.set_yticks([])
ax.set_ylabel("cells, by genotype")
fig.colorbar(im, ax=ax, label="smoothed log ratio")
```

## Notes

- Count matrices with float dtype are fine as long as the values are whole
  numbers. Normalised data is rejected.
- `min_cells` (default 50) is the smallest pseudobulk. On a sample with few
  cells, lower it through `params=cnv.CnvParams(min_cells=...)`.
- Stage timings go to stdout with `verbose=True`. In Jupyter they appear in
  the terminal running the kernel, not in the notebook.
