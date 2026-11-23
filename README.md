# cnv-sc-rs

Copy number inference from single-cell expression. A Rust port of the
expression path of [numbat](https://github.com/kharchenkolab/numbat) and its
numerics package hahmmr (Gao et al., *Nature Biotechnology* 41, 417-426, 2023),
both MIT.

## What it does

Tier 0, the inferCNV capability. Counts, gene coordinates and a reference go in:

1. Genomic order, exclusion regions, reference choice per cell.
2. Sparse fused smoothing to bins (an inferCNV-style signal).
3. Cell hierarchy by kNN-Ward (via `bonsai-rs`), cut into nested groups.
4. Pseudobulk per group, then a 15-state Poisson-lognormal HMM and Viterbi.
5. Segment retest and consensus segments across groups.
6. Per-cell segment posteriors and a tree-free clone posterior.

One pass, no tree-based iteration. [SPEC](docs/SPEC.md) section 10 has the
full list, with numbat file and line for every piece.

## What it isn't

- No I/O. Plain slices in, plain structs out.
- No allele data. Tier 1 adds it, so the loh/bamp/bdel states are folded into
  neu/amp/del for now (SPEC 11, D5).
- No clone phylogeny. Tier 2. Until then the clone posterior runs over leaf
  genotypes (D8).
- No GPU path. On 10 threads the per-cell stage runs at 0.3 ms per cell, so
  there is nothing yet for a GPU to win.

## Usage

```rust
use cnv_sc_rs::cnv::run_cnv;
use cnv_sc_rs::genome::CsrCounts;
use cnv_sc_rs::prelude::*;
use cnv_sc_rs::reference::Reference;

// indptr, indices, data: cells x genes CSR, u32. chrom, start, end: per gene.
let counts = CsrCounts::new(&indptr, &indices, &data, n_genes)?;
// Indices of the cells you trust to be normal, pooled into one profile.
let reference = Reference::<f32>::Cells(&normal_cells);

let res = run_cnv::<f32>(&counts, &chrom, &start, &end, &reference, None, Verbosity::Quiet)?;

for seg in &res.consensus {
    println!("{seg:?}");
}
```

`None` takes numbat's defaults; tune via `CnvParams`. `CnvResult` holds the
smoothed bins, the groups, the consensus segments, per-cell posteriors
(`None` if every segment is neutral), entropy and confidence per segment, and
the clone posterior.

## Inputs

- CSR counts, `u32` indices and counts, caller gene order.
- Per gene: chromosome id (`u32`), start and end (`u64`).
- A reference: `Reference::Cells` (indices of normal cells, pooled),
  `Reference::Groups` (normal cells with a label each, one profile per label)
  or `Reference::Profiles` (genes x `n_ref` proportions, column-major). With
  more than one profile, each cell is compared against the one it correlates
  best with.
- Optional exclusion regions in `CnvParams::exclude`, e.g. the MHC.

Storage is generic over `CnvFloat`, default `f32`. Every reduction and
loglikelihood accumulates in `f64`. Same input gives the same output at any
thread count.

## Streaming

`run_cnv` takes anything that implements `CellBlocks`, not just a matrix in
memory. The pipeline reads the cells in four passes, block by block, in
ascending order, and holds one block of counts at a time. `CsrCounts`
implements it without copying. For an on-disk store, implement `with_block`
to read cells `lo..hi` into your own buffers. The result doesn't depend on
`CnvParams::block_cells` (default 8192).

## Performance

Simulated data, Apple Silicon laptop, 10 threads, f32, 10k genes. Full
numbers and the tuning measurements are in [PERFORMANCE](docs/PERFORMANCE.md).

| cells | `run_cnv` total | peak heap above the input |
|---|---|---|
| 10k | 3.83 s | 58 MB |
| 20k | 7.55 s | 103 MB |
| 50k | 18.43 s | 204 MB |

The per-cell posterior step is most of it. It went from 65.6 s to 3.0 s at 10k
cells through a method-of-moments start and Chebyshev interpolants for small
counts. numbat's own per-cell fit takes 1137 ms on one thread against 2.05 ms
here.

infercnvpy 0.6.1 on the same matrices takes 24.4 s, 48.1 s and 172.0 s in
total. That isn't like for like: it has no HMM, segments or per-cell
probabilities, and its `infercnv` step alone is faster than our whole run at
20k and 50k (5.54 s and 7.17 s). It peaks at 1.95, 3.15 and 6.28 GB RSS.

```bash
cargo bench --bench pipeline
CNV_CELLS=50000 CNV_GENES=10000 CNV_COPY=1 cargo bench --bench pipeline
```

`CNV_CELLS` and `CNV_GENES` set the size (default 10k x 10k). `CNV_COPY=1`
copies every block into fresh buffers, as a store would. `CNV_DUMP=<dir>`
writes the counts, genes and truth as text so other tools can run on the same
matrix.

## Validation

Expected values come from running R, never from memory. The scripts in
`reference/` call hahmmr and `source()` the needed numbat files, and write
small TSV fixtures to `reference/fixtures/`. The tests compare against them:
PLN density and fit, HMM transitions and Viterbi, reference choice, smoothing,
segments and per-cell posteriors. Deviations from numbat are listed in
[SPEC](docs/SPEC.md) section 11.

## Licence

MIT. See `LICENSE`. numbat and hahmmr are MIT and ported directly; their
licences are in `LICENSES/`. [Provenance](PROVENANCE.md) has what was read and
under which licence. Tier 2 will be written from the paper only.

## Citing

Cite Gao et al., 2023. This is an independent port and claims none of the
science.
