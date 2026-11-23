# cnv-sc-rs

Copy number inference from single cell expression, in Rust, with a thin
Python layer on top.

It is a port of the expression path of
[numbat](https://github.com/kharchenkolab/numbat) (Gao et al., *Nature
Biotechnology* 2023) and its numerics package hahmmr, both MIT:

1. A smoothed log-ratio signal per cell against a set of normal reference
   cells, binned along the genome (the inferCNV view).
2. A cell hierarchy on that signal, cut into pseudobulks.
3. A Poisson-lognormal HMM per pseudobulk, then consensus segments across
   pseudobulks, retested for a gain, loss or LOH call.
4. Per-cell posteriors for every non-neutral segment and a tree-free tumour
   probability per cell.

No allele data and no clone phylogeny yet; those are later tiers of the
Rust crate.

## Install

```bash
pip install cnv-sc-rs              # numpy, scipy, polars
pip install "cnv-sc-rs[anndata]"   # plus infer_cnv_adata
```

Wheels are abi3, so one wheel covers Python 3.10 onwards.

## What you need

- **Raw UMI counts**, cells x genes, as a scipy sparse matrix. Not
  normalised, not log-transformed.
- **Gene loci**: chromosome, start and end per gene, on the genome build
  your annotation uses. Genes off chromosomes 1 to 22 are dropped, as in
  numbat.
- **Normal reference cells**: cells you are confident carry no CNV, for
  instance immune or stromal cells from the same sample. Several cell types?
  Label them and each gets its own profile.

Start with the [quickstart](quickstart.md), or go to the
[scanpy guide](scanpy.md) if your data lives in an `AnnData`.
