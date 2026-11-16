# Orchestrator handover (paused 2026-11-16, usage limit)

Resume after 21:00 when limits reset. Plan: `dev/plan_tier0.md`. Spec:
`docs/SPEC.md`. Rules: `CLAUDE.md`, `PROVENANCE.md`.

## State

- Branch `worktree-dapper-prancing-teacup` (orchestrator worktree
  `/Users/gregor/repos/shared/cnv-sc-rs/.claude/worktrees/dapper-prancing-teacup`).
- Commits on top of `main` (`da8362f`):
  - `1baa118`: scaffold, SPEC, provenance, CI
  - `f77aaf7`: B and C handovers
  - plus this handover
- M0 is done. `cargo test --release` passes 4 tests and clippy is clean.
- The R oracle is installed in `reference/.rlib` (gitignored): hahmmr,
  poilog, roptim. numbat is not installed (it imports scistreer).
- Measured: hahmmr and `poilog::dpoilog` differ by about 4e-7 relative at
  (0, 1, 0.5).

## Workstreams: sequential, one subagent at a time (user request)

| ws | handover | status |
|---|---|---|
| A emissions | `dev/handover-A-emissions.md` | done, cherry-picked (`7503003..f9965c0`). GH s/56 nodes, ~540 ns/call; D9 no floor |
| B hmm + segments | `dev/handover-B-hmm-segments.md` | done, ff-merged (`d7416b1`, `26ae36d`); D10 added, SPEC 6.5/D5 fixed in `fc00469`; Viterbi 93 ns/gene, emissions 2.7 us/gene |
| C reference + smoothing | `dev/handover-C-reference-smoothing.md` | done, ff-merged (`0faca1a`, `06553c5`); smoothing 213k cells/s 1 thread, 1.29M cells/s 10 threads at G=10k |

## A at pause

- Branch `worktree-agent-a2be01602b992f925` at
  `/Users/gregor/repos/shared/cnv-sc-rs/.claude/worktrees/agent-a2be01602b992f925`,
  one commit `1a3da1a wip: workstream A pause`. Status is in
  `dev/status-A-emissions.md` there.
- Done: `pln.rs`, `optim.rs`, fixtures. 21 of 21 lib tests pass.
  `GH_NODES = 56`, max relative error 4.2e-7 on held-out points against a
  tight R reference. `fit_lnpois` is >= hahmmr on all sets. `phi` agrees
  with numbat to 4e-6.
- Left: bench numbers, delete `tmp_sweep_nodes`, clippy, squash the WIP
  commit.
- Questions for Gregor before merge:
  1. Floor. SPEC 4.1 says `ln p >= ln 1e-15`, but hahmmr only replaces an
     underflowed `p == 0`, and 55 of 175 grid points sit between the two.
     That is my transcription error in SPEC. Recommendation: no floor, since
     the log-density never underflows. Fix SPEC.
  2. Quadrature scale `0.8*s` with 32 nodes against the SPEC `s` with 56
     nodes. 0.8 scaled is more accurate on the grid (5.5e-7 held-out) and
     cheaper, so adopt it once the bench confirms. The scale is ours, not
     upstream.
  3. `phi = 2.5` (`l32`) is computed but unused in numbat; leave it out.

## M2 done (2026-11-16)

- `cluster.rs` (`4d5dea1`): top-down cut of bonsai `linkage_tree` by between-children SS (Ward cost); bonsai sets branches to 1, so heights are recomputed lazily.
- `posterior.rs` (`ab15122`, parity `27ff23b`): per cell x non-neu segment f32 log-odds [logBF, loh, del, amp]; clone posterior uses only these (Z_n sum cancels). Parity vs numbat get_exp_likelihoods + compute_posterior: mu/sigma 2.2e-6, log-odds 1.3e-5 rel.
- `cnv.rs` (`6913bbb`, `ea87290`): `run_cnv`, stage timings at Verbosity::Normal. Simulation (600 cells, 3000 genes, 2 clones): exact segment recovery, tumour/normal 1.000, clone 1.000.
- Stage split on that small case (10 threads): reference 0.00 s, smoothing 0.00, clustering 0.01, bulk HMMs 0.09, consensus 0.10, cell posteriors 1.24 s.

## Next

- M3: pick a public dataset with known CNVs with Gregor; infercnvpy comparison.
- M4: stage split at a larger size (report before scaling); target is the per-cell PLN stage.
- Open decisions for Gregor: GH quadrature 0.8*s/40 nodes (22% faster, more accurate) vs SPEC s/56; LOH fold under D5.

## Resume steps (historical)

1. `git worktree list` to find A's worktree. Read its
   `dev/status-A-emissions.md` and `git log`.
2. Resume A. Either send a message to the paused agent if it still exists,
   or start a fresh general-purpose agent in a worktree based on A's branch.
   Give it the A handover plus the status file.
3. Review A's numbers, then merge A into this branch (fast-forward or
   merge). Run `cargo test --release`.
4. Run B, then C, the same way: one at a time, base equals current branch tip.
5. Then M2 (mine): `cluster.rs`, `posterior.rs`, `cnv.rs`. Check that
   `bonsai_rs::tree::linkage::linkage_tree` and `tree::cluster::cluster` are
   public in bonsai-rs 0.3.1.

## Open with Gregor

- M3 dataset with known CNVs: not chosen yet.
- Merge gotcha: the untracked `dev/cnv_project_start.md` in the main
  checkout blocks a fast-forward merge of this branch.
