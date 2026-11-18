# Handover A: Poisson-lognormal emissions

Read first: `CLAUDE.md`, `PROVENANCE.md`, `docs/SPEC.md` sections 0 and 4.
Also read the rust-style and cpu-optimisation skills.

## Scope

1. `src/model/pln.rs`, covering SPEC 4.1 to 4.5:
   - `ln_pln(x: u32, mu: f64, sigma: f64) -> f64`: log-density, floored at `ln 1e-15`.
   - A fused variant returning the log-density plus the derivatives in
     `(mu, sigma)` (first and second) from the same quadrature pass.
   - `l_lnpois` over a gene set: slices `y: &[u32]`, `lambda: &[T]`, `d: f64`,
     `mu`, `sigma`, `phi`.
   - A multi-phi variant that evaluates several `phi` values in one pass
     over the genes. SPEC 8.4 needs `{1, 0.5, 1.5, 2, 0.25}`.
   - `fit_lnpois`: bounded Newton, `sigma >= 0.01`, start `(0, 1)`.
   - `phi_mle` on `[0.1, 10]` and `approx_phi_post` (Laplace, `[0.2, 10]`).
2. Any small optimiser you need goes in `src/utils/optim.rs`. 1-D bounded
   only; keep it minimal.
3. Errors go in the "emission models" section of `src/errors.rs`.

## Oracles

Write `reference/pln.R` (see CLAUDE.md "R oracle"; the library is already
installed in the main worktree at
`/Users/gregor/repos/shared/cnv-sc-rs/.claude/worktrees/dapper-prancing-teacup/reference/.rlib`.
Point `R_LIBS` at that absolute path). Fixtures:

- Density grid. Use `x` in {0, 1, 2, 5, 20, 100, 1000}, `mu` in {-8, -3, 0, 2, 6}
  and `sigma` in {0.01, 0.1, 0.5, 1, 2}. For each point record:
  - a tight reference from `integrate(..., rel.tol = 1e-12)` on the
    integrand in SPEC 4.1;
  - `poilog::dpoilog`;
  - `hahmmr:::dpoilog`.
- `l_lnpois` and `fit_lnpois_cpp` on a few simulated gene sets (seeded
  `rpois(exp(rnorm))` draws with lambda from a Dirichlet-like draw, 500 to
  5000 genes, zeros dominant).
- `approx_phi_post` and `calc_phi_mle_lnpois` from numbat (`source()`
  `~/repos/others/numbat/R/utils.R`, which may need a few library calls),
  on the same sets with a planted `phi` of 0.5, 1 and 1.5.

## Acceptance

- Relative error against the tight reference is <= 1e-6 across the grid.
  The Gauss-Hermite node count is a named const; its doc comment gives the
  measured max error and the date.
- `fit_lnpois` objective is >= hahmmr's - 1e-6 * |hahmmr's|; report the
  parameter deltas.
- Laplace `phi_hat` and `s_phi` agree with numbat to 1e-4 relative, or
  explain the gap.
- Single-thread throughput bench `benches/pln.rs`: ns per `ln_pln` call and
  per fused call, with zero-count genes dominant. Report the numbers;
  do not optimise beyond the obvious yet.
- `cargo test --release` green, clippy clean.

## Out of scope

HMM, segments, reference, smoothing, GPU. Do not edit `docs/SPEC.md`
except to add a deviation to section 11 if one turns out to be needed, and
flag it in the report.

## Report back

1. What was built, with file paths.
2. Measured accuracy and timings.
3. Any SPEC ambiguity found.
4. The commits on your branch.
