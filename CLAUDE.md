# CLAUDE.md

Guidance for Claude Code in this repository.

## Crate

`cnv-sc-rs`: copy number inference from single cell expression. A port of
the expression path of numbat and hahmmr (both MIT). Library only, numerics
only, no I/O. Inputs are plain slices: CSR counts (`u32`), per-gene
chromosome and position, and a reference. No bixverse or store types.

Tier 0 (current) is the inferCNV capability. Tier 1 adds alleles. Tier 2 adds
the clone phylogeny. Plan and handovers are in `dev/`.

## Licence rules, binding

**Never open, install, read or grep scistreer** (GPL-3), its source or its
installed R package. Not here, not in a sibling worktree, not through a
subagent. Do not install numbat into any R library, because it pulls in
scistreer. See [PROVENANCE.md](PROVENANCE.md).

numbat and hahmmr are MIT: read and port them freely. Implementation follows
[docs/SPEC.md](docs/SPEC.md). Cite `SPEC x.y` in doc comments. A deviation
from numbat goes into SPEC section 11 first.

## Commands

```bash
cargo build --release
cargo test --release
cargo test --release -- model::pln::tests::test_name --exact --nocapture
cargo bench --bench <name>
```

## R oracle

Isolated R library at `reference/.rlib` (gitignored): hahmmr from
`~/repos/others/hahmmr`, plus `poilog` and `roptim`. Set it up once:

```bash
mkdir -p reference/.rlib && printf 'FLIBS=\n' > reference/.rlib/Makevars
R_LIBS=reference/.rlib Rscript -e 'install.packages(c("roptim","poilog"), lib="reference/.rlib", repos="https://cloud.r-project.org")'
R_MAKEVARS_USER=$PWD/reference/.rlib/Makevars R CMD INSTALL --library=reference/.rlib ~/repos/others/hahmmr
```

The `FLIBS=` override works around missing gfortran libs on this Mac.
Fixture scripts live in `reference/<topic>.R`. Run them with
`R_LIBS=reference/.rlib Rscript reference/<topic>.R`. They write to
`reference/fixtures/` (committed, small, plain TSV). numbat functions are
`source()`d from `~/repos/others/numbat/R/*.R`. **Expected values always come
from running R, never from memory.**

## Numerics

- Storage generic over `CnvFloat` (a `BonsaiFloat`), default `f32`. Every
  reduction and loglikelihood accumulates in `f64`.
- Loglikelihoods are exact (include `lgamma(x+1)`), not up to a constant.
- Determinism: same input, same output, any thread count. Never `.sum()` on a
  `ParallelIterator`; reduce per unit in a fixed order.

## Conventions

- British English. `#![warn(missing_docs)]`; every item gets `### Params` and
  `### Returns`.
- One `thiserror` enum in `src/errors.rs`; add to the matching section.
- Tuning constants are named `const`s whose doc comment gives the source: a
  numbat/hahmmr file:line, or our measurement with the date.
- `wide` types only in `src/utils/simd.rs`. No SIMD tier without a
  measurement in its doc comment.
- Benches: plain `main`, `harness = false`. Check the output before reporting
  a timing. Measure a stage's share of the whole before optimising it.
