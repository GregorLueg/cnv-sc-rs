# Performance and tuning measurements

Every tuning constant in the code is chosen from a measurement recorded here.
Machine: Apple Silicon laptop, 10 threads. All dates 2026-11-16.

## Pipeline (`benches/pipeline.rs`)

Simulated 10k cells x 10k genes, 21.8% nonzero, f32, 10 threads. Seconds per
stage.

| stage | initial | + moment start + count interpolants |
|---|---|---|
| reference | 0.01 | 0.01 |
| smoothing | 0.03 | 0.03 |
| clustering | 0.53 | 0.48 |
| bulk HMMs | 0.26 | 0.19 |
| consensus | 0.17 | 0.12 |
| cell posteriors | 65.60 | 3.00 |
| total | 66.6 | 3.83 |

Peak heap above the 174 MB input: 53 to 58 MB.

### Per-cell posterior step, one thread

1000 cells of the same simulation. numbat: its own `get_exp_likelihoods` on
50 cells with our consensus segments (9660 genes, 439 in non-neutral
segments).

| | baseline fit | segment likelihoods | total |
|---|---|---|---|
| numbat | 1137 ms | about 43 ms | 1180 ms |
| ours, initial | 50.3 ms (6.9 Newton iterations) | 1.19 ms | 51.5 ms |
| + moment start | 21.7 ms (2.7 iterations) | 1.19 ms | about 23 ms |
| + count interpolants | 2.05 ms | 0.28 ms | 2.35 ms |

The f32 `mu`, `sigma` and segment log-odds of all 1000 cells are
bit-identical with and without the interpolants. The moment start moves the
fit by at most 1.5e-6, within `FIT_TOL`.

### infercnvpy 0.6.1 on the same matrix

Given the true normal cells as reference. `infercnv` 4.6 s, then PCA,
neighbours, Leiden and score 80.5 s (first call, may include compilation),
peak RSS 1.68 GB. No HMM, segments or per-cell probabilities, so not like for
like.

## PLN quadrature (`GH_NODES`)

Max relative error in `p` against R `integrate` at `rel.tol = 1e-12`, on
`pln_grid.tsv` (175 points, SPEC grid) / `pln_random.tsv` (400 held-out
points, sigma up to 2.5):

| nodes | grid | held-out |
|---|---|---|
| 32 | 3.0e-6 | 7.0e-6 |
| 40 | 8.6e-7 | 2.6e-6 |
| 48 | 2.6e-7 | 1.1e-6 |
| 56 | 8.7e-8 | 4.2e-7 |
| 64 | 3.2e-8 | 9.8e-8 |

The worst points are always `x <= 2` with `sigma >= 2`. 56 is the smallest
count under 1e-6 on both sets.

Nodes narrowed to `0.8 s` (`benches/pln.rs`): 32 nodes 1.8e-7 / 5.5e-7 at
360 ns per `ln_pln` call, 40 nodes 3.7e-8 / 4.8e-8 at 419 ns, against 536 ns
for 56 nodes at `s`. Not adopted: SPEC 4.1 fixes `s`.

hahmmr and `poilog::dpoilog` differ by about 4e-7 relative at
`(x, mu, sigma) = (0, 1, 0.5)`.

## Count interpolants (`src/model/pln.rs`)

Fit per cell, one thread, 1000 cells x 10k genes, degree 16, by the largest
tabled count (`MAX_TABLED_COUNT`):

| tabled counts | fit per cell |
|---|---|
| none | 21.7 ms |
| 0 | 6.16 ms |
| up to 1 | 3.34 ms |
| up to 3 | 2.40 ms |
| up to 6 | 2.33 ms |

Interpolation error against `ln_pln_fused` over `t` in `[-12, 4]`, counts 0
to 3, `sigma` 0.01 to 3, worst relative to `max(1, |f|)`, for `ln p` /
`d_sigma` / `d2_sigma`:

| degree, width | `ln p` | `d_sigma` | `d2_sigma` |
|---|---|---|---|
| 16, 1 | 3e-15 | 1.8e-13 | 8.0e-11 |
| 16, 2 | 3.7e-13 | 5.5e-13 | 7.1e-11 |
| 16, 4 | 6.0e-9 | 3.5e-8 | 1.9e-6 |
| 12, 2 | 1.7e-10 | 3.1e-9 | 7.3e-8 |
| 10, 2 | 3.4e-9 | 2.4e-7 | 1.3e-6 |

Degree 12 at width 2 cut the fit from 2.40 to 2.05 ms per cell against
degree 16, with bit-identical f32 posteriors.

## Smoothing (`benches/smooth.rs`)

G = 10k genes, 5% density: 213k cells/s on one thread (about 5 us per cell),
1.29M cells/s on 10 threads. `CELL_MIN_LEN` = 64 is not swept.
