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
| A emissions | `dev/handover-A-emissions.md` | paused mid-run; WIP committed on its own worktree branch (see `dev/status-A-emissions.md` there) |
| B hmm + segments | `dev/handover-B-hmm-segments.md` | not started, needs A merged |
| C reference + smoothing | `dev/handover-C-reference-smoothing.md` | not started, independent of A and B |

## Resume steps

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
