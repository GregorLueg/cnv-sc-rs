# Proposal: CNV inference (inferCNV replacement, numbat-style)

Status: discussion, nothing decided or built. Written 2026-11-10.

inferCNV is deprecated. The goal is one CNV stack in Rust that covers the
inferCNV use case (count matrix only) and numbat's allele-aware model, so the
same code serves 10x RNA, ATAC tiles and allele counts.

Read from:

- numbat @ `b444f3c` (1.5.2), local checkout in `~/repos/others/numbat`.
  hahmmr and scistreer were not read, only their imports in numbat's
  `NAMESPACE`. Licences of both unchecked; numbat is MIT.
- bixverse-rs main @ `e449453` (0.10.0), `main == origin/main`.

Storage questions (genome index, allele store, ATAC) live in
[`bixverse-bundle.md`](bixverse-bundle.md).

## 1. What numbat does

About 7k lines of R, mostly dplyr orchestration. The numerics sit in two
Kharchenko lab packages:

- **hahmmr:** `l_lnpois` / `fit_lnpois_cpp` (Poisson-lognormal expression
  likelihood against a reference profile), `dbbinom` / `l_bbinom`
  (beta-binomial on phased het SNP counts), `run_allele_hmm_s5`,
  `run_joint_hmm_s15`, `forward_back_allele`.
- **scistreer:** `perform_nni`, `score_tree`, `mut_to_tree`. Max-likelihood
  phylogeny by NNI.

The loop in `run_numbat` (`R/main.R`): cluster cells on smoothed expression,
pseudobulk per cluster, run the joint HMM on the pseudobulks, build consensus
segments, compute per-cell posteriors per segment, build the tree, call clones,
repeat with the clones as the new groups. Phase switch probability between
SNPs comes from genetic distance (`switch_prob_cm`).

Inputs on top of the count matrix: BAMs, a cellsnp-lite pileup and Eagle2
phasing against a 1000G panel and genetic map (`inst/bin/pileup_and_phase.R`).
That buys copy-neutral LOH and a tumour/normal split that doesn't need a
diploid reference cell type. The paper claims it beats inferCNV and CopyKAT;
that's their benchmark, not checked here. No BAMs, no numbat.

**numbat already contains inferCNV.** `smooth_expression` (`R/utils.R`) is the
inferCNV core: log2 CPM ratio against the reference, cap at +-3, centre per
cell, running mean over 101 genes. `exp_hclust` (`R/main.R`) then does Ward
clustering on it. numbat has an `allele_only` switch on its group HMMs but no
expression-only one.

Matrix-only alternatives for comparison: infercnvpy (Python port of the
inferCNV core), CopyKAT, SCEVAN. XClone is numbat's peer (alleles plus
expression). Maintenance status of all four unchecked.

## 2. Tiers

| tier | inputs | gives you |
|---|---|---|
| 0 | counts, gene coordinates, reference (named normal cells or a reference profile) | inferCNV-style heatmap, clusters, expression-only HMM, per-cell segment posteriors |
| 1 | + sparse AD/DP and phased SNPs | joint HMM, allele-specific calls, LOH, better tumour/normal split |
| 2 | output of 0 or 1 | clone phylogeny via NNI |

Tier 0 needs no genomics I/O and ships first. Its HMM has to be designed, not
ported. numbat bundles HCA-derived reference profiles; those are data and go
in the R layer, not the crate. Support both reference modes, it's cheap.

CNV from ATAC is tier 0 or 1 on tiles instead of genes (numbat's multiome mode,
`inst/bin/get_binned_atac.R`). The model shouldn't care where the bins came
from.

## 3. Where the code lives

Same split as `bonsai-rs`:

1. **Separate crate, numerics only, no I/O.** Emission models
   (Poisson-lognormal, beta-binomial), HMM engine generic over the emission,
   per-cell posteriors from segment counts, NNI tree search. Callable from
   Python without bixverse.
2. **`bixverse-sc`, behind a `cnv` feature.** Genome index use, the streaming
   smoothing and binning pass, segment aggregation, the iterative loop.
3. **R side** on `SingleCells`. Whether that is bixverse proper or its own
   package (`bixverse.cnv`, like `bixverse.gpu`) is open.

Not inside bixverse-rs proper: it brings genomic coordinates, SNP tables and
genetic maps that nothing else in the workspace uses, and the nine crates are
versioned together.

## 4. Tier 0 on the streaming reader

Generic over `S: SingleCellReading`.

| step | access | call |
|---|---|---|
| reference profile, gene filter | gene-major, reference cells | `read_genes_sparse(genes, Some(ref_cells), Raw)` |
| library sizes | cheap | `read_cell_library_sizes` |
| log ratio, cap, centre, smooth along the genome | per cell, genes in genomic order | `read_cells_sparse` in `CELL_BATCH_SIZE` batches plus a genome-order permutation |
| cluster pseudobulks | exists | `get_pseudo_bulked_counts_sparse` |
| HMM on pseudobulks | small, in memory | none |
| per-cell segment counts | one cell-major pass, gene columns summed into segments | `read_cells_sparse` |

**The smoothed matrix is dense.** 1M cells x 10k genes in f32 is 40 GB. The
pass must reduce on the fly to fixed genomic bins (~100 genes, so 100 to 200
bins per cell, ~0.8 GB at 1M cells). Clustering, PCA and the heatmap run off
the bins. Bin width against resolution on small focal events is unmeasured;
settle it on a dataset with known CNVs.

## 5. Allele inputs

1. **Now:** cellsnp-lite (Gregor's accelerated fork) and Eagle2 stay external
   CLI steps. The crate reads cellsnp-lite's native output (sparse AD/DP/OTH
   mtx plus the base VCF) and the phased genotypes, not numbat's long
   `df_allele`. That contract also decouples the crate from the pileup and
   phasing tools.
2. **Later, if timings say so:** a Rust pileup at known SNP sites (noodles,
   rayon over regions), i.e. cellsnp-lite mode 1a.
3. **Never:** reimplement population phasing. Call Eagle2, SHAPEIT5 or Beagle.

Reference-panel phasing is human only. Mouse, short of F1 hybrids, has almost
no het SNPs, so it falls back to tier 0.

## Open

1. Which cellsnp-lite the input contract targets: upstream or the company fork.
2. Which numbat stages dominate runtime. Unmeasured; one sample with per-stage
   timing settles what to port first.
3. R home: bixverse proper or a separate package.
4. hahmmr and scistreer licences.
