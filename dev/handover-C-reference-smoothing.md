# Handover C: genome, reference and smoothed bins

Read first: `CLAUDE.md`, `PROVENANCE.md`, and `docs/SPEC.md` sections 0 to 3
and 11. Also read the rust-style and cpu-optimisation skills.

## Scope

1. `src/genome.rs` (SPEC 1):
   - `GenomeOrder` built from `chrom: &[u32]`, `start: &[u64]`, `end: &[u64]`;
   - the sort permutation, `chrom_ranges`, exclusion intervals, kept-gene
     masks.
2. `src/reference.rs` (SPEC 2):
   - `Reference::{Profiles, Cells}`, `aggregate`, `filter_genes`;
   - `choose_ref_cor`, sparse, rayon over cells;
   - `lambdas_bar`;
   - `fit_ref_sse` with the analytic gradient and lower bounds. Write a
     minimal projected quasi-Newton or projected gradient yourself; no new
     dependency.
3. `src/smooth.rs` (SPEC 3):
   - the sparse fused evaluation (3.3) producing dense cells x bins
     (row-major, `T: CnvFloat`);
   - a dense naive path for tests only;
   - parallel over cell chunks with `for_each_init` scratch;
   - library size over kept genes.

   Input CSR is cells x genes:
   - `indptr: &[u32]`, `indices: &[u32]`, `data: &[u32]`;
   - genes in caller order, mapped through `GenomeOrder`.
4. Errors go in the "genome" and "reference" sections.

## Oracles

`reference/reference.R` and `reference/smooth.R`. The R library is at
`/Users/gregor/repos/shared/cnv-sc-rs/.claude/worktrees/dapper-prancing-teacup/reference/.rlib`.
`source()` numbat `R/utils.R` from `~/repos/others/numbat` (never install
numbat). `caTools` is in the system library.

1. `filter_genes`, `choose_ref_cor`, `get_lambdas_bar` and `fit_ref_sse` on a
   simulated count matrix: 300 cells x 2000 genes over 4 chromosomes, with
   2 or 3 reference profiles.
2. `smooth_expression` on the same data.
   - numbat smooths across chromosome boundaries (D1). Build the parity
     fixture by calling numbat's transform, then `caTools::runmean` per
     chromosome in the R script, so it encodes our deviation explicitly.
   - Also keep one fixture from unmodified `smooth_expression`. With a
     single chromosome, ours must match it exactly.
3. Bins: average the R-side smoothed matrix over bins in R (SPEC 3.2 rules)
   for the bin fixture.

## Acceptance

1. Gene sets and reference choices are identical.
2. `fit_ref_sse` objective is <= R's * (1 + 1e-6).
3. Smoothed bins match to 1e-5 absolute in f64, and to a stated tolerance
   in f32.
4. Sparse fused equals dense naive to 1e-10 in f64.
5. Bench `benches/smooth.rs`: cells per second for the bins pass at
   G = 10k, 5% density. Report single-thread and all-thread numbers, and
   peak extra memory.
6. `cargo test --release` green, clippy clean.

## Out of scope

HMM, segments, emissions, clustering, GPU. SPEC edits only to add a
deviation, flagged in the report.

## Report back

Built paths, measurements, SPEC ambiguities, branch and commits.
