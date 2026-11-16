//! Single-thread throughput of the PLN density (SPEC 4).
//!
//! Inputs mimic one cell against a reference: `mu` spread like
//! `mu0 + ln(d lambda)`, counts drawn from the PLN itself, so zeros dominate.

use std::hint::black_box;
use std::time::Instant;

use cnv_sc_rs::model::pln::{fit_lnpois, l_lnpois_multi, ln_pln, ln_pln_fused};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Poisson};

/// Number of `(x, mu)` pairs per pass.
const N: usize = 200_000;
/// Timed repeats; the minimum is reported.
const REPEATS: usize = 7;
/// Latent standard deviation used throughout.
const SIGMA: f64 = 0.8;

/// Best-of-`REPEATS` wall time of `f`, in ns per item.
fn time_ns<F: FnMut() -> f64>(mut f: F, items: usize) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPEATS {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed().as_nanos() as f64 / items as f64);
    }
    best
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let mu_dist = Normal::new(-1.5, 2.0).unwrap();
    let z_dist = Normal::new(0.0, SIGMA).unwrap();
    let mut x = Vec::with_capacity(N);
    let mut mu = Vec::with_capacity(N);
    for _ in 0..N {
        let m: f64 = mu_dist.sample(&mut rng);
        let rate = (m + z_dist.sample(&mut rng)).exp();
        x.push(Poisson::new(rate).unwrap().sample(&mut rng) as u32);
        mu.push(m);
    }
    let zero_frac = x.iter().filter(|&&v| v == 0).count() as f64 / N as f64;
    println!("{N} pairs, zero fraction {zero_frac:.3}, sigma {SIGMA}");

    let ns = time_ns(
        || (0..N).map(|i| ln_pln(black_box(x[i]), mu[i], SIGMA)).sum(),
        N,
    );
    println!("ln_pln        {ns:7.1} ns/call");

    let ns = time_ns(
        || {
            (0..N)
                .map(|i| ln_pln_fused(black_box(x[i]), mu[i], SIGMA).d2_sigma)
                .sum()
        },
        N,
    );
    println!("ln_pln_fused  {ns:7.1} ns/call");

    for (label, keep) in [("x == 0", true), ("x > 0", false)] {
        let idx: Vec<usize> = (0..N).filter(|&i| (x[i] == 0) == keep).collect();
        let ns = time_ns(
            || {
                idx.iter()
                    .map(|&i| ln_pln(black_box(x[i]), mu[i], SIGMA))
                    .sum()
            },
            idx.len(),
        );
        println!("ln_pln {label:7}{ns:7.1} ns/call ({} calls)", idx.len());
    }

    // gene-set level: lambda with d * lambda = exp(mu + 1.5)
    let d = 10_000.0;
    let lambda: Vec<f32> = mu.iter().map(|m| ((m + 1.5).exp() / d) as f32).collect();
    let phis = [1.0, 0.5, 1.5, 2.0, 0.25];
    let ns = time_ns(
        || l_lnpois_multi(&x, &lambda, d, -1.5, SIGMA, &phis).unwrap()[0],
        N * phis.len(),
    );
    println!("l_lnpois_multi (5 phi) {ns:7.1} ns per gene and phi");

    let g = 5000;
    let ns = time_ns(|| fit_lnpois(&x[..g], &lambda[..g], d).unwrap().mu, 1);
    let fit = fit_lnpois(&x[..g], &lambda[..g], d).unwrap();
    println!(
        "fit_lnpois {g} genes {:.3} ms ({} iterations, mu {:.4}, sigma {:.4})",
        ns / 1e6,
        fit.iterations,
        fit.mu,
        fit.sigma
    );
}
