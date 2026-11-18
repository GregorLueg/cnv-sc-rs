# cnv-sc-rs: tier 0 (inferCNV capability) — plan

## Context

`dev/cnv_project_start.md` proposes one Rust CNV stack to replace inferCNV and
grow into numbat. The first release is tier 0: counts + gene coordinates +
reference gives smoothed CNV signal, a cell hierarchy, an expression HMM on
pseudobulks, consensus segments, per-cell segment posteriors and a tree-free
tumour/normal call. It must be a standalone crate (plain slices in, no bixverse
or v5 types), laid out like bonsai-rs, fast and lean on memory, with a GPU path
only where timings justify one.

What mapping found (sources: numbat @ b444f3c, hahmmr local, both MIT):
- numbat has no working expression-only mode. `exp_only` skips the retest, so
  every segment ends up `neu` (numbat R/utils.R:591, R/main.R:890), and
  `run_group_hmms` drops bulks without allele data (main.R:805). We rebuild the
  expression-only retest from numbat's Laplace pieces (`approx_phi_post`,
  utils.R:1508).
- Tumour/normal only exists through the scistreer tree (`get_clone_post`,
  main.R:1022). We feed the same formula with leaf-cluster genotypes instead.
- Hot spot: `get_exp_post` (main.R:1330). It fits one Poisson-lognormal per
  cell and evaluates about 30 x G_seg PLN integrals per cell by adaptive QAGS
  quadrature. Second: the dense G x C smoothing plus O(C^2) Ward.
- hahmmr is MIT, so a direct port is fine. scistreer is GPL-3 and belongs to
  tier 2, which is deferred.

Decisions taken:
1. Cell hierarchy: bonsai-rs kNN-Ward (`tree::linkage_tree`, `tree::cluster`).
2. Fidelity: parity on the kernels, fix the pipeline. Deviations are listed in
   SPEC.
3. Tree-free clone posterior is included.
4. Scope: the crate plus a GPU path. The bixverse adapter and Python bindings
   come later.

## Crate layout (bonsai-rs pattern)

```
Cargo.toml       edition 2024, MIT; deps: bonsai-rs (linkage only), rayon, num-traits,
                 thiserror, wide, rustc-hash; optional cubecl (feature `gpu`, `gpu-tests`)
src/lib.rs       policy header, #![warn(missing_docs)], VERSION
src/prelude.rs
src/errors.rs    single CnvErrors enum, sectioned
src/cnv.rs       pipeline: run_cnv(..) -> CnvResult (one pass, no tree)
src/genome.rs    GenomeOrder: per-gene chrom (u32) + start/end (u64), genomic sort perm,
                 chromosome ranges, exclusion intervals (MHC passed in, no build constants)
src/reference.rs filter_genes, choose_ref_cor, fit_ref_sse, lambdas_bar, aggregate
src/smooth.rs    log-ratio / cap / centre / per-chromosome running mean, fused into bins
src/cluster.rs   bins -> bonsai linkage_tree -> nested groups (2k-1), pseudobulk sums
src/model/pln.rs Poisson-lognormal: log-density, fused grad/hess, fit_lnpois, phi MLE/Laplace
src/hmm/         engine (log-space Viterbi), 15-state expression model, transitions
src/segments.rs  annot/smooth segs, retest, resolve_cnvs (union-find), fill_neu, multi-allelic
src/posterior.rs per-cell fit + segment likelihoods + compute_posterior; clone posterior
src/utils/       traits.rs (CnvFloat, as BonsaiFloat), simd.rs (only `wide` file),
                 lse.rs, optim.rs (small projected Newton / quasi-Newton, no deps)
src/gpu/         (feature gpu) per-cell PLN kernel, only after M4 timing
benches/         plain main, harness=false, one per stage + pipeline
reference/       R oracle scripts producing fixtures (see Oracles)
docs/            SPEC.md (implementation source), DESIGN.md, PERFORMANCE.md
dev/             project start doc, handovers per workstream
PROVENANCE.md, CLAUDE.md, .github/workflows/test.yml (+ release.yml) calling
GregorLueg/personal-actions@v1 like bonsai-rs
```

Inputs are plain: CSR cells x genes with `u32` indptr/indices and `u32`
counts; `GenomeOrder`; `Reference::{Cells(Vec<usize>), Profiles{lambdas, n_refs}}`.
Storage is generic over `CnvFloat` (default f32). Every reduction runs in f64.

## Key numerics and performance design (all unmeasured until benched)

- **Sparse fused smoothing.** A zero count gives a cell-independent value
  `c_g = clamp(-log2(lambda_g*1e6+1), -3, 3)`. Centring, running mean and
  binning are all linear, so a cell's bins are `bin(smooth(c))` (computed
  once) plus the contribution of its nonzeros. Each nonzero's correction
  `delta_g` spreads into at most a few bins through precomputed gene-to-bin
  weights. Cost per cell drops from O(G) to O(nnz), and nothing dense of size
  G x C ever exists. Library size is taken over kept genes, as numbat does.
  Checked against a dense naive path in tests.
- **PLN density.** Newton finds the mode of the concave log-integrand, then
  Gauss-Hermite runs around it with Laplace scaling. The node count is set by
  measuring accuracy against `poilog::dpoilog`, target 1e-6 relative. The
  value, gradient in (mu, sigma) and Hessian come from the same node pass
  (posterior moments of z), which gives analytic derivatives where hahmmr
  uses numerical ones.
- **fit_lnpois.** Bounded 2-D Newton (sigma >= 0.01). Parity is checked on
  the final objective and the parameters against hahmmr's `fit_lnpois_cpp`.
- **HMM.** Only 5 distinct phi exist, so 5 emission columns are computed and
  mapped to 15 states. In expression-only mode the transition matrix is
  constant. Viterbi runs in log space, in parallel over (group, chromosome).
  Forward-backward waits for tier 1.
- **Per-cell posteriors.** Rayon over cells with `for_each_init` scratch. The
  segment likelihoods at 6 phi values are fused into one pass over each
  segment's genes. Zero-count genes dominate; the zero-count density as a 1-D
  function of the shifted mu is a candidate table, applied only if M4 shows it
  pays.
- **GPU.** The per-cell PLN stage is the obvious candidate: embarrassingly
  parallel and heavy on exp. It gets built only if the M4 stage split shows it
  dominates wall-clock. Uses the cubecl-learnings skill.

## Milestones and who does what

**M0, orchestrator (me).**
1. Copy the start doc into `dev/`.
2. Scaffold the crate, errors, traits/simd/lse/optim and CI.
3. Write `docs/SPEC.md` from the mapping, with numbat/hahmmr file:line
   citations and a deviations section.
4. Write `PROVENANCE.md` and `CLAUDE.md`.
5. Set up the oracle environment and fixtures.
6. Commit. This is the base every workstream branches from.

**M1, three workstreams, run sequentially (A, then B, then C) to keep usage
down.** One general-purpose subagent at a time (model inherited), in its own
worktree off the current branch tip, given SPEC sections plus
`dev/handover-<ws>.md`. Each is merged before the next starts. None of them
spawns further agents.

| ws | scope | parity target |
|---|---|---|
| A emissions | `model/pln.rs`, `utils/optim.rs` use, Laplace phi, phi MLE | `poilog::dpoilog`, hahmmr `l_lnpois`, `fit_lnpois_cpp`; numbat `approx_phi_post`, `calc_phi_mle_lnpois` |
| B hmm + segments | `hmm/`, `segments.rs` | hahmmr Viterbi on fixed log-emissions; numbat `annot_segs`, `smooth_segs`, `resolve_cnvs`, `fill_neu_segs`, `test_multi_allelic` on fixture tables |
| C reference + smoothing | `genome.rs`, `reference.rs`, `smooth.rs` | numbat `filter_genes`, `choose_ref_cor`, `fit_ref_sse`; `smooth_expression` with a per-chromosome deviation test |

**M2, orchestrator.** Merge A, B and C. Build `cluster.rs` (bonsai linkage on
bins), `posterior.rs` (per-cell plus clone posterior) and `cnv.rs`
end-to-end.

**M3, correctness.**
1. Simulate counts from PLN with planted gains, losses and normal cells, then
   check segment recovery and tumour/normal accuracy.
2. Compare against infercnvpy 0.6.1 (available through uv) on a public
   dataset with known CNVs. The dataset is chosen with you.

**M4, performance.**
1. Time the stage split on the small case and report back.
2. Then optimise the dominant stage only.
3. Benchmarks against infercnvpy wall-clock and peak RSS.

**M5, GPU.** A cubecl kernel for the per-cell PLN stage, only if M4 says so.
Feature-gated, with CPU/GPU parity tests.

**Tier 2 (later).** A clean-room scistreer from the Bioinformatics 36(3):742
paper. One session writes SPEC from the paper, a different one implements from
SPEC, as bonsai-rs did. bonsai's `LiveTree` and NNI move machinery is reusable;
its Brownian scoring is not.

## Oracles

- An isolated R library at `reference/.rlib` (gitignored) holds hahmmr
  installed from `~/repos/others/hahmmr` plus `poilog` from CRAN. **numbat is
  not installed**, because it imports scistreer. Fixture scripts `source()`
  the needed numbat R files from the MIT checkout instead.
- `reference/*.R` writes small fixtures (TSV or binary) into
  `reference/fixtures/`. Rust tests read them. Every expected value comes from
  running R, never from memory.

## Provenance rules (go into PROVENANCE.md and CLAUDE.md)

- numbat and hahmmr are MIT. Port them, attribute them, keep their LICENSE text.
- scistreer is GPL-3. No session opens, installs, reads or greps its source or
  its installed R package, directly or through a subagent. Tier 2 works from
  the paper only.

## Verification

1. `cargo test --release` (CI: `--release`; gpu: `--release --features gpu-tests`).
2. Kernel parity tests against the R fixtures, with tolerances stated per test.
3. Dense-versus-sparse smoothing equivalence test.
4. Simulation recovery test (M3) and the infercnvpy comparison (M3).
5. `cargo bench --bench pipeline` and per-stage benches, small case first, then
   report before scaling up.

## Handback

Work stays on `worktree-dapper-prancing-teacup`, and workstream branches merge
into it. I never push.

Merge gotcha: `dev/cnv_project_start.md` is untracked in the main checkout. The
branch commits a copy at the same path, so a `git merge --ff-only` on main will
refuse until the untracked original is removed or moved.
