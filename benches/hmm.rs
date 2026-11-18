//! Single-thread throughput of the expression HMM (SPEC 5): Viterbi alone on
//! fixed log-emissions, and the emission build that feeds it.

use std::hint::black_box;
use std::time::Instant;

use cnv_sc_rs::hmm::{N_STATES, emission_log_probs, log_initial, log_transitions, viterbi};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Gamma, Normal, Poisson};

/// Genes per sequence, about one large chromosome.
const N: usize = 5_000;
/// Timed repeats; the minimum is reported.
const REPEATS: usize = 9;

/// Best-of-`REPEATS` wall time of `f`, in ns per gene.
fn time_ns<F: FnMut() -> usize>(mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_nanos() as f64 / N as f64);
    }
    best
}

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let g = Gamma::new(0.6, 1.0).unwrap();
    let z: Normal<f64> = Normal::new(0.0, 0.4).unwrap();
    let mut lambda: Vec<f64> = (0..N).map(|_| g.sample(&mut rng)).collect();
    let s: f64 = lambda.iter().sum();
    lambda.iter_mut().for_each(|l| *l *= 0.2 / s);
    let d = 1e6;
    let y: Vec<u32> = lambda
        .iter()
        .map(|&l| {
            let rate: f64 = z.sample(&mut rng).exp() * d * l;
            Poisson::new(rate.max(1e-12)).unwrap().sample(&mut rng) as u32
        })
        .collect();

    let log_a = log_transitions(1e-5).unwrap();
    let log_d = log_initial(1e-5).unwrap();

    let e = emission_log_probs(&y, &lambda, d, 0.0, 0.4, 0.25);
    let ns = time_ns(|| emission_log_probs(black_box(&y), &lambda, d, 0.0, 0.4, 0.25).len());
    println!("emissions  {ns:8.1} ns/gene (5 PLN evaluations)");

    let ns = time_ns(|| {
        viterbi(&log_d, &log_a, black_box(&e), N_STATES)
            .unwrap()
            .len()
    });
    println!("viterbi    {ns:8.1} ns/gene (M = {N_STATES})");
}
