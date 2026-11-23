# Handover B: HMM engine and segments

Read first: `CLAUDE.md`, `PROVENANCE.md`, and `docs/SPEC.md` sections 0, 5, 6
and 11. Also read the rust-style and cpu-optimisation skills. Workstream A
(`src/model/pln.rs`) is merged. Use its `fit_lnpois`, `l_lnpois` and
`approx_phi_post`; do not re-implement them.

## Scope

1. `src/hmm/` (SPEC 5):
   - State table.
   - Transition and initial log-matrices from `(t, w, logphi_min)`.
   - Log-space Viterbi over a `n x M` row-major log-emission slice.
   - Emission builder for a pseudobulk:
     - inputs `y: &[u32]`, `lambda`, `d`, `mu`, `sigma`;
     - 5 distinct phi columns computed once, then mapped to 15 states.
   - Parallel over independent sequences (per chromosome, per bulk) with
     rayon. A single Viterbi stays sequential.
2. `src/segments.rs` (SPEC 6):
   - `annot_segs`, `smooth_segs` (D3);
   - expression-only retest (6.4, D4, D5);
   - consensus with union-find components (6.5);
   - `fill_neu`, retest on consensus (6.6), multi-state (6.7).

   Segment types are plain structs:
   - chromosome id, gene index range `[lo, hi)` in genomic order, bp
     start/end, state, LLR, LLR_x, `p_*`, `phi_hat`, `s_phi`;
   - a bulk id where relevant.
3. Errors go in the "hmm and segments" section.

## Oracles

`reference/hmm.R` and `reference/segments.R`. The R library is at
`/Users/gregor/repos/shared/cnv-sc-rs/.claude/worktrees/dapper-prancing-teacup/reference/.rlib`.

- **Transitions.** hahmmr's `calc_trans_mat_s15` and `get_trans_probs_s15`
  output for the defaults and for `t = 1e-3`. Compare entries, including
  the -Inf ones.
- **Viterbi.** hahmmr's `viterbi_compute` on fixed random log-emissions (a
  few sequences, n = 50 to 2000, M = 15). The state paths must be identical.
  Also an end-to-end `run_joint_hmm_s15(exp_only = TRUE)` on a simulated
  pseudobulk with planted gain and loss; the states must be identical.
- **Segments.** `source()` numbat `R/utils.R` and `R/main.R` from
  `~/repos/others/numbat` (never install numbat). Run `annot_segs`,
  `smooth_segs`, `resolve_cnvs`, `fill_neu_segs` and `test_multi_allelic` on
  small hand-built tables. Run the expression-only retest maths by calling
  numbat's `retest_cnv` pieces (`approx_phi_post`, `pnorm.range.log`) with
  `p_s = 0` and no alleles. Where a deviation (D3 to D5) applies, test our
  behaviour separately and say so in the test name.

## Acceptance

1. Paths and transitions match hahmmr exactly (log-probabilities to 1e-12).
2. Segment tables match numbat field for field on the fixtures, apart from
   the documented deviations.
3. Bench `benches/hmm.rs`: Viterbi ns per gene, M = 15.
4. `cargo test --release` green, clippy clean.

## Out of scope

Per-cell posteriors, clustering, reference, smoothing, GPU. SPEC edits only
to add a deviation, flagged in the report.

## Report back

Built paths, measurements, SPEC ambiguities, branch and commits.
