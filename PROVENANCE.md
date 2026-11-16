# Provenance

What was read and under which licence. This is the basis for the MIT licence
on this crate.

## Sources used

| source | licence | used for |
|---|---|---|
| numbat @ `b444f3c` (1.5.2), Teng Gao et al. | MIT, [LICENSES/numbat.md](LICENSES/numbat.md) | tier 0 pipeline, smoothing, reference handling, retest, consensus, per-cell posteriors |
| hahmmr (local checkout, 2023), hahmmr authors | MIT, [LICENSES/hahmmr.md](LICENSES/hahmmr.md) | Poisson-lognormal density and fit, 15-state HMM, Viterbi |
| Gao et al., *Haplotype-aware analysis of somatic copy number variations from single-cell transcriptomes*, Nature Biotechnology 41, 417-426 (2023) | paper | model background |

Both packages are MIT, so this crate ports them directly. [docs/SPEC.md](docs/SPEC.md)
cites file and line for each piece. Deviations are listed in SPEC section 11.

## scistreer: off limits

scistreer is **GPL-3**. A port would be a derivative work and could not be
MIT.

No session on this repository opens, installs, reads or greps scistreer's
source or its installed R package. That covers direct access, a sibling
worktree and a subagent. numbat imports scistreer, so the R oracle in
`reference/` does not install numbat. It `source()`s the specific MIT R files
it needs from the numbat checkout instead.

Tier 2 (clone phylogeny) will be written from the paper only: Gao et al.,
*Bioinformatics* 36(3):742 (2020). One session transcribes the maths into a
spec, and a different session implements from that spec, as in bonsai-rs.

Pretraining caveat: scistreer is public, so it may be in a model's training
data. The answer is the same as for a human who once saw the code. The
implementation is written from the spec, and this file records how.
