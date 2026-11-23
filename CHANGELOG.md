# Changelog

## 0.1.0 (unreleased)

First version. A port of the expression path of numbat and hahmmr, both MIT;
see [provenance](PROVENANCE.md).

**Model**

- Poisson-lognormal density, fit and `phi` posterior, with Laplace-centred
  Gauss-Hermite quadrature and analytic derivatives.
- 15-state expression HMM, transitions and Viterbi.
- Genome order, CSR counts, reference profiles and reference choice per cell
  (SPEC 1-2).
- Sparse fused smoothing to bins, per-chromosome window (SPEC 3).
- Segment annotation, expression-only retest and consensus (SPEC 6).
- Nested cell groups from a top-down cut of the kNN-Ward tree (`bonsai-rs`).
- Per-cell segment posteriors and a tree-free clone posterior (SPEC 8, 9).

**Pipeline**

- `cnv::run_cnv`: one pass, tier 0, with `CnvParams` and `CnvResult`.
- `Reference::Groups`: labelled normal cells, one profile per label. The
  Python `infer_cnv` takes `reference_labels` or precomputed profiles, and
  `infer_cnv_adata` takes `reference_group_key`. Pooled results unchanged.
- `CellBlocks`: cells are read in four block passes from any source. `CsrCounts`
  implements it without copying.
- `benches/pipeline.rs`: stage split, wall-clock and peak heap on simulated
  counts, with `CNV_CELLS`, `CNV_GENES`, `CNV_COPY` and `CNV_DUMP`.

**Performance**

- Method-of-moments start for `fit_lnpois` and Chebyshev interpolants for PLN
  sums over small counts. Per-cell posteriors at 10k cells x 10k genes went
  from 65.6 s to 3.0 s; f32 posteriors are bit-identical with and without the
  interpolants. See [performance](docs/PERFORMANCE.md).

**Numerics**

- Storage generic over `CnvFloat`, reductions in `f64`.
- Deterministic whatever the thread count.
- Parity against numbat and hahmmr through R fixtures in `reference/`.
- Duplicate gene indices within a cell are rejected.
