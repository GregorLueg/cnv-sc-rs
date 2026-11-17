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

10k genes, 10 threads/jobs. infercnvpy gets the true normal cells as
reference and an untimed warm-up on 1000 cells, so compilation is excluded
(the first cold run at 10k took 85 s). It has no HMM, segments or per-cell
probabilities, so this is not like for like. Peak RSS is the whole process:
ours includes the simulation and the text dump, infercnvpy's the warm-up.

| cells | ours total | infercnvpy `infercnv` | infercnvpy PCA to score | infercnvpy total | ours peak RSS | infercnvpy peak RSS |
|---|---|---|---|---|---|---|
| 10k | 3.83 s | 4.87 s | 19.5 s | 24.4 s | 0.73 GB | 1.95 GB |
| 20k | 7.55 s | 5.54 s | 42.6 s | 48.1 s | 1.45 GB | 3.15 GB |
| 50k | 18.43 s | 7.17 s | 164.8 s | 172.0 s | 3.34 GB | 6.28 GB |

At 20k our per-stage seconds: smoothing 0.06, clustering 1.06, bulk HMMs
0.21, consensus 0.11, cell posteriors 6.12; peak heap above the 349 MB input
103 MB. Tumour/normal accuracy 0.966 (10k) and 0.965 (20k).

### GSE148673 (`examples/gse148673.rs`)

Gao et al. (CopyKat) UMI matrices; copykat `N` cells as the reference.
TNBC1: 1,097 cells, 19,755 autosomal expressed genes (GENCODE v19), 20.6%
nonzero. ATC2: 6,226 cells, 22,400 genes (Ensembl 93), 9.1% nonzero. 10
threads, 2026-11-17.

| data | baseline total | posteriors | + Chebyshev moments total | posteriors | + SIMD node masses total | posteriors |
|---|---|---|---|---|---|---|
| TNBC1 | 1.63 s | 1.40 s | 1.41 s | 1.00 s | 0.99 s | 0.70 s |
| ATC2 | 5.42 s | 4.47 s | 3.16 s | 2.25 s | 2.46 s | 1.71 s |
| simulation 10k | 3.87 s | 3.08 s | 2.80 s | 2.00 s | 2.20 s | 1.49 s |
| simulation 50k | 18.80 s | 15.28 s | 13.59 s | 10.06 s | 11.62 s | 7.96 s |

Posterior stage profile at baseline, one thread, macOS `sample`, share of
the stage:

| | TNBC1 | ATC2 |
|---|---|---|
| `sum_grouped` own time (per-gene Clenshaw) | 19% | 50% |
| `ln_pln_fused`, all callers | 56% | 38% |
| of which libm `exp` + `expm1` | 36% | 25% |
| `FusedInterp::new` (table build) | 33% | 22% |

Chebyshev moments (`ChebMoments`): per-panel sums of `T_k(u_g)` once per
fit instead of a Clenshaw pass per gene and evaluation, and no table for an
empty panel. f32 posteriors bit-identical on TNBC1; on ATC2 2 of 74,712
log-odds differ, by at most 1.5e-8.

SIMD node masses (`utils::simd::pln_node_masses`): the 56-node loop of
`pln_quadrature` in `f64x4` with an in-house `exp`/`expm1` (range
reduction, degree-13 Taylor, no special cases; within 2.5e-16 relative of
libm). f32 posteriors unchanged against the moments-only run. `benches/pln.rs`
on aarch64, ns per call:

| | `ln_pln` | `ln_pln_fused` |
|---|---|---|
| scalar libm | 547 | 626 |
| `wide` `exp` / `exp_m1` | 1000 | 1075 |
| in-house, `round` + `splat` in the loop | 813 to 977 | 885 to 1065 |
| in-house, `const` vectors, shift-constant rounding | 344 | 434 |

Why the middle rows lose: `wide`'s `exp` pays for NaN, overflow and
subnormal handling per call; `f64x2::round` has no NEON path (scalar `as u64`
through an array); `f64x4::splat` inside the loop compiled to a
`memset_pattern16` call per constant and iteration. x86-64 is not measured;
without target features `f64x4` is SSE2 pairs.

### Reading cells in blocks (`CellBlocks`)

50k cells x 10k genes, 10 threads, `BLOCK_CELLS` = 8192. In memory (no
copy): total 18.84 s. Every block copied into fresh buffers, as a store
would: 19.06 s. Peak heap above the input 204 MB and 196 MB. Before blocks
(one pass over the whole matrix): 18.43 s.

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

A group is tabled only when its evaluations exceed the interpolant's node
count. On the 1000-cell case (segments of 200 and 250 genes) this changed
nothing: fit 2.06 ms, segment likelihoods 0.28 ms per cell, `mu` and `sigma`
bit-identical. The case it guards, small segments with a wide `t` spread, is
not measured.

## Smoothing (`benches/smooth.rs`)

G = 10k genes, 5% density: 213k cells/s on one thread (about 5 us per cell),
1.29M cells/s on 10 threads. `CELL_MIN_LEN` = 64 is not swept.
