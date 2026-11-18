//! Throughput of the fused smoothing-to-bins pass (SPEC 3.3) at G = 10k
//! genes, 5% density: cells per second on one thread and on all threads,
//! and the peak heap on top of the output.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use cnv_sc_rs::genome::{CsrCounts, GenomeOrder};
use cnv_sc_rs::smooth::smooth_bins;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};

/// Heap allocator that tracks live and peak bytes.
struct Counting;

/// Live heap bytes.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// Peak of `LIVE` since the last reset.
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
        PEAK.fetch_max(now, Ordering::Relaxed);
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Genes.
const G: usize = 10_000;
/// Chromosomes, equal size.
const N_CHROM: usize = 20;
/// Nonzeros per cell, 5% density.
const NNZ: usize = G / 20;
/// Cells.
const C: usize = 20_000;
/// Timed repeats; the minimum is reported.
const REPEATS: usize = 5;

fn main() {
    let mut rng = StdRng::seed_from_u64(3);
    let gamma = Gamma::new(0.5, 1.0).unwrap();
    let chrom: Vec<u32> = (0..G).map(|g| (g / (G / N_CHROM)) as u32 + 1).collect();
    let start: Vec<u64> = (0..G).map(|g| 1_000 * g as u64).collect();
    let order = GenomeOrder::new(&chrom, &start, &start).unwrap();
    let mut lambda: Vec<f64> = (0..G).map(|_| gamma.sample(&mut rng) + 1e-3).collect();
    let s: f64 = lambda.iter().sum();
    lambda.iter_mut().for_each(|l| *l /= s);
    let keep = vec![true; G];

    let mut indptr = vec![0u32];
    let mut indices: Vec<u32> = Vec::with_capacity(C * NNZ);
    let mut data: Vec<u32> = Vec::with_capacity(C * NNZ);
    let mut row: Vec<u32> = Vec::with_capacity(NNZ);
    for _ in 0..C {
        row.clear();
        while row.len() < NNZ {
            let g = rng.random_range(0..G as u32);
            if !row.contains(&g) {
                row.push(g);
            }
        }
        row.sort_unstable();
        indices.extend_from_slice(&row);
        data.extend((0..NNZ).map(|_| 1 + rng.random_range(0..4u32)));
        indptr.push(indices.len() as u32);
    }
    let counts = CsrCounts::new(&indptr, &indices, &data, G).unwrap();

    let run = || {
        let mut best = f64::INFINITY;
        let mut extra = 0usize;
        for _ in 0..REPEATS {
            let base = LIVE.load(Ordering::Relaxed);
            PEAK.store(base, Ordering::Relaxed);
            let t = Instant::now();
            let out = smooth_bins::<f32>(black_box(&counts), &order, &keep, &lambda, None).unwrap();
            best = best.min(t.elapsed().as_secs_f64());
            let out_bytes = out.data.capacity() * 4;
            extra = PEAK.load(Ordering::Relaxed) - base - out_bytes;
            black_box(out);
        }
        (C as f64 / best, best, extra)
    };

    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let (cps1, t1, ex1) = one.install(run);
    let n_all = rayon::current_num_threads();
    let (cps, t, ex) = run();
    println!(
        "G = {G}, C = {C}, nnz/cell = {NNZ}, f32 output, bins = {}",
        G / 100
    );
    println!(
        "1 thread   {cps1:12.0} cells/s  ({:.1} ms)  extra heap {:.2} MB",
        t1 * 1e3,
        ex1 as f64 / 1e6
    );
    println!(
        "{n_all:2} threads {cps:12.0} cells/s  ({:.1} ms)  extra heap {:.2} MB",
        t * 1e3,
        ex as f64 / 1e6
    );
}
