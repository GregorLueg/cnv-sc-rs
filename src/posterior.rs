//! Per-cell segment posteriors (SPEC 8) and the tree-free clone posterior
//! (SPEC 9, D8).
//!
//! Per cell and non-neutral consensus segment the pass stores four `f32`
//! log-odds against neutral: the pooled `logBF = Z_cnv - Z_n` and
//! `Z_x - Z_n` for `x` in loh, del, amp. That is all the clone posterior
//! needs: its per-genotype score is `sum_s Z_n(c, s)` plus log-odds of the
//! genotype's segments, and the first sum is the same for every genotype, so
//! it cancels in the softmax. Memory is `16 * n_cells * n_segments` bytes.

use rayon::prelude::*;

use crate::errors::CnvErrors;
use crate::genome::{CsrCounts, GenomeOrder};
use crate::model::pln::{fit_lnpois, l_lnpois_multi};
use crate::reference::Profiles;
use crate::segments::{PostState, Segment};
use crate::utils::lse::log_sum_exp;

/// Segment state priors below this are set to 0 (N/R/main.R:1400-1403).
const PRIOR_FLOOR: f64 = 0.05;

/// The `phi` values of SPEC 8.4, in the order `l11, l10, l21, l31, l00`.
const POST_PHIS: [f64; 5] = [1.0, 0.5, 1.5, 2.0, 0.25];

/// Log-odds stored per cell and segment.
pub const N_LOG_ODDS: usize = 4;

/// Column of the pooled `logBF` in [`CellPosteriors::log_odds`].
pub const COL_LOGBF: usize = 0;
/// Column of `Z_loh - Z_n`.
pub const COL_LOH: usize = 1;
/// Column of `Z_del - Z_n`.
pub const COL_DEL: usize = 2;
/// Column of `Z_amp - Z_n`.
pub const COL_AMP: usize = 3;

/// Per-cell posteriors over the non-neutral consensus segments.
#[derive(Clone, Debug)]
pub struct CellPosteriors {
    /// Caller cell index per row
    pub cells: Vec<u32>,
    /// Index into the consensus of each non-neutral segment (columns)
    pub segs: Vec<usize>,
    /// `cells x segs x N_LOG_ODDS`, row-major; `NaN` for a cell whose
    /// baseline could not be fitted
    pub log_odds: Vec<f32>,
    /// Baseline `mu` per cell
    pub mu: Vec<f32>,
    /// Baseline `sigma` per cell
    pub sigma: Vec<f32>,
}

impl CellPosteriors {
    /// Log-odds of one cell and segment column.
    ///
    /// ### Params
    ///
    /// * `row` - Cell row
    /// * `col` - Segment column
    ///
    /// ### Returns
    ///
    /// `[logBF, loh, del, amp]` against neutral.
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> &[f32] {
        let i = (row * self.segs.len() + col) * N_LOG_ODDS;
        &self.log_odds[i..i + N_LOG_ODDS]
    }

    /// Posterior probability of a CNV, `p_cnv = 1 / (1 + exp(-logBF))`.
    ///
    /// ### Params
    ///
    /// * `row` - Cell row
    /// * `col` - Segment column
    ///
    /// ### Returns
    ///
    /// `p_cnv`, `NaN` for an unfitted cell.
    #[inline]
    pub fn p_cnv(&self, row: usize, col: usize) -> f64 {
        1.0 / (1.0 + (-(self.at(row, col)[COL_LOGBF] as f64)).exp())
    }

    /// Mean binary entropy of `p_cnv` per segment (SPEC 8.7), over fitted
    /// cells, `0 * log 0 = 0`.
    ///
    /// ### Returns
    ///
    /// Entropy in bits per segment column.
    pub fn entropy(&self) -> Vec<f64> {
        let n_seg = self.segs.len();
        (0..n_seg)
            .into_par_iter()
            .map(|s| {
                let (mut sum, mut k) = (0.0f64, 0usize);
                for c in 0..self.cells.len() {
                    let p = self.p_cnv(c, s);
                    if p.is_nan() {
                        continue;
                    }
                    let h = |q: f64| if q > 0.0 { -q * q.log2() } else { 0.0 };
                    sum += h(p) + h(1.0 - p);
                    k += 1;
                }
                if k == 0 { f64::NAN } else { sum / k as f64 }
            })
            .collect()
    }
}

/// The state a consensus segment carries into the posteriors: its retest
/// state, `neu` if untested.
///
/// ### Params
///
/// * `s` - Consensus segment
///
/// ### Returns
///
/// The effective state.
pub fn effective_state(s: &Segment) -> PostState {
    s.retest.map_or(PostState::Neu, |r| r.state_post)
}

/// Genes per segment column and baseline genes, for one reference profile.
struct RefGenes {
    /// Genomic gene indices of the baseline (neutral) genes with `lambda > 0`
    base: Vec<u32>,
    /// Their `lambda`
    base_lambda: Vec<f64>,
    /// Per segment column, genomic gene indices with `lambda > 0`
    seg: Vec<Vec<u32>>,
    /// Their `lambda`
    seg_lambda: Vec<Vec<f64>>,
    /// Every gene counted into `D_c`: in some consensus segment, `lambda > 0`
    all: Vec<u32>,
}

/// Per-thread scratch of the cell pass.
struct Scratch {
    /// Dense counts in genomic order, zero outside the current cell
    y: Vec<u32>,
    /// Gathered counts of one gene set
    ys: Vec<u32>,
}

/// Per-cell posteriors over the non-neutral consensus segments (SPEC 8).
///
/// Per cell, with its reference profile: `D_c` over the genes in consensus
/// segments with `lambda > 0`; `(mu, sigma)` by `fit_lnpois` on genes of
/// neutral segments (plus LOH-labelled ones with `use_loh`); then each
/// non-neutral segment's loglikelihood at the five `phi` of SPEC 8.4 in one
/// pass, combined with the segment's priors (floored at
/// [`PRIOR_FLOOR`]). Rayon over cells.
///
/// ### Params
///
/// * `counts` - Count matrix, caller gene order
/// * `order` - Genomic order of the genes
/// * `profiles` - Reference profiles, genomic order
/// * `cells` - Cells to score
/// * `refs` - Reference profile per entry of `cells`
/// * `consensus` - Consensus segments with retest and (multi-state) priors
/// * `use_loh` - Whether LOH-labelled neutral segments join the baseline
///
/// ### Returns
///
/// The posteriors. A cell with `D_c = 0` or a failed baseline fit gets `NaN`
/// rows rather than failing the run.
pub fn cell_posteriors(
    counts: &CsrCounts,
    order: &GenomeOrder,
    profiles: &Profiles,
    cells: &[u32],
    refs: &[u32],
    consensus: &[Segment],
    use_loh: bool,
) -> Result<CellPosteriors, CnvErrors> {
    if refs.len() != cells.len() {
        return Err(CnvErrors::ShapeMismatch {
            what: "refs",
            expected: cells.len(),
            got: refs.len(),
        });
    }
    counts.check_cells(cells)?;
    let n_genes = order.n_genes();
    let segs: Vec<usize> = (0..consensus.len())
        .filter(|&i| effective_state(&consensus[i]) != PostState::Neu)
        .collect();
    let n_seg = segs.len();

    // priors per segment column, in the order (loh, del, amp, bamp, bdel)
    let priors: Vec<[f64; 5]> = segs
        .iter()
        .map(|&i| {
            let r = consensus[i]
                .retest
                .expect("non-neutral segments are retested");
            [r.p_loh, r.p_del, r.p_amp, r.p_bamp, r.p_bdel]
                .map(|p| if p < PRIOR_FLOOR { 0.0 } else { p })
        })
        .collect();

    let ref_genes: Vec<RefGenes> = (0..profiles.n_ref)
        .map(|r| {
            let lam = profiles.profile(r);
            let mut g = RefGenes {
                base: Vec::new(),
                base_lambda: Vec::new(),
                seg: vec![Vec::new(); n_seg],
                seg_lambda: vec![Vec::new(); n_seg],
                all: Vec::new(),
            };
            let mut col = 0;
            // a gene belongs to its first overlapping segment (N/R/main.R:1262)
            let mut taken = vec![false; n_genes];
            for (i, s) in consensus.iter().enumerate() {
                let is_col = col < n_seg && segs[col] == i;
                let is_base =
                    effective_state(s) == PostState::Neu && (use_loh || !s.state.is_loh());
                for gene in s.lo..s.hi {
                    if std::mem::replace(&mut taken[gene as usize], true) {
                        continue;
                    }
                    let l = lam[gene as usize];
                    if l <= 0.0 {
                        continue;
                    }
                    g.all.push(gene);
                    if is_col {
                        g.seg[col].push(gene);
                        g.seg_lambda[col].push(l);
                    } else if is_base {
                        g.base.push(gene);
                        g.base_lambda.push(l);
                    }
                }
                if is_col {
                    col += 1;
                }
            }
            g
        })
        .collect();

    let mut log_odds = vec![0.0f32; cells.len() * n_seg * N_LOG_ODDS];
    let mut mu = vec![f32::NAN; cells.len()];
    let mut sigma = vec![f32::NAN; cells.len()];
    let row_len = n_seg * N_LOG_ODDS;
    let rows: Vec<(usize, &mut [f32])> = if row_len == 0 {
        Vec::new()
    } else {
        log_odds.chunks_exact_mut(row_len).enumerate().collect()
    };
    let fits: Vec<(f32, f32)> = rows
        .into_par_iter()
        .map_init(
            || Scratch {
                y: vec![0u32; n_genes],
                ys: Vec::new(),
            },
            |sc, (row, out)| {
                let c = cells[row] as usize;
                let rg = &ref_genes[refs[row] as usize];
                let (idx, val) = counts.row(c);
                for (&j, &v) in idx.iter().zip(val) {
                    sc.y[order.rank[j as usize] as usize] = v;
                }
                let d: f64 = rg.all.iter().map(|&g| sc.y[g as usize] as f64).sum();
                let res = score_cell(sc, rg, &priors, d, out);
                for &j in idx {
                    sc.y[order.rank[j as usize] as usize] = 0;
                }
                if res.is_none() {
                    out.fill(f32::NAN);
                }
                res.unwrap_or((f32::NAN, f32::NAN))
            },
        )
        .collect();
    for (i, (m, s)) in fits.into_iter().enumerate() {
        mu[i] = m;
        sigma[i] = s;
    }
    Ok(CellPosteriors {
        cells: cells.to_vec(),
        segs,
        log_odds,
        mu,
        sigma,
    })
}

/// Score one cell into its output row (SPEC 8.3 to 8.6).
///
/// ### Params
///
/// * `sc` - Scratch with the cell's dense counts loaded
/// * `rg` - Gene sets of the cell's reference
/// * `priors` - Floored priors per segment column, `(loh, del, amp, bamp, bdel)`
/// * `d` - `D_c`
/// * `out` - Row of `n_seg * N_LOG_ODDS` log-odds
///
/// ### Returns
///
/// `(mu, sigma)`, or `None` if `D_c = 0` or a fit or loglikelihood failed.
fn score_cell(
    sc: &mut Scratch,
    rg: &RefGenes,
    priors: &[[f64; 5]],
    d: f64,
    out: &mut [f32],
) -> Option<(f32, f32)> {
    if d <= 0.0 || rg.base.is_empty() {
        return None;
    }
    sc.ys.clear();
    sc.ys.extend(rg.base.iter().map(|&g| sc.y[g as usize]));
    let fit = fit_lnpois(&sc.ys, &rg.base_lambda, d).ok()?;
    let ln_half = 0.5f64.ln();
    for (s, pi) in priors.iter().enumerate() {
        sc.ys.clear();
        sc.ys.extend(rg.seg[s].iter().map(|&g| sc.y[g as usize]));
        let l = l_lnpois_multi(&sc.ys, &rg.seg_lambda[s], d, fit.mu, fit.sigma, &POST_PHIS).ok()?;
        let (l11, l10, l21, l31, l00) = (l[0], l[1], l[2], l[3], l[4]);
        let [p_loh, p_del, p_amp, p_bamp, p_bdel] = *pi;
        // N/R/main.R:1415-1442; ln(0) = -inf drops a state
        let z_n = l11 + ln_half;
        let z_loh = l11 + (p_loh / 2.0).ln();
        let z_del = l10 + (p_del / 2.0).ln();
        let z_amp = log_sum_exp(&[l21 + (p_amp / 4.0).ln(), l31 + (p_amp / 4.0).ln()]);
        let z_bamp = l31 + (p_bamp / 2.0).ln();
        let z_bdel = l00 + (p_bdel / 2.0).ln();
        let z_cnv = log_sum_exp(&[z_loh, z_del, z_amp, z_bamp, z_bdel]);
        let o = &mut out[s * N_LOG_ODDS..(s + 1) * N_LOG_ODDS];
        o[COL_LOGBF] = (z_cnv - z_n) as f32;
        o[COL_LOH] = (z_loh - z_n) as f32;
        o[COL_DEL] = (z_del - z_n) as f32;
        o[COL_AMP] = (z_amp - z_n) as f32;
    }
    Some((fit.mu as f32, fit.sigma as f32))
}

/////////////////////
// Clone posterior //
/////////////////////

/// One CNV in a genotype.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenotypeEvent {
    /// Segment column in [`CellPosteriors`]
    pub col: usize,
    /// State for a multi-state segment; `None` uses the pooled `logBF`
    pub state: Option<PostState>,
}

/// Clone assignment per cell (SPEC 9).
#[derive(Clone, Debug)]
pub struct ClonePosterior {
    /// Genotypes; index 0 is the normal (empty) genotype
    pub genotypes: Vec<Vec<GenotypeEvent>>,
    /// `cells x genotypes` posterior, row-major
    pub prob: Vec<f32>,
    /// Most probable genotype per cell
    pub clone: Vec<u32>,
    /// `1 - p(normal)` per cell
    pub p_tumour: Vec<f32>,
}

/// Tree-free clone posterior (SPEC 9, D8).
///
/// Prior 0.5 on the normal genotype and `0.5 / (n - 1)` on each other;
/// score `ln prior + sum` of the genotype's log-odds; softmax per cell.
/// Unfitted cells get `NaN` probabilities and clone 0.
///
/// ### Params
///
/// * `post` - Per-cell posteriors
/// * `genotypes` - Non-empty genotypes; duplicates and an empty one are
///   dropped, the normal genotype is added at index 0
///
/// ### Returns
///
/// The clone posterior.
pub fn clone_posterior(post: &CellPosteriors, genotypes: &[Vec<GenotypeEvent>]) -> ClonePosterior {
    let mut gts: Vec<Vec<GenotypeEvent>> = vec![Vec::new()];
    for g in genotypes {
        let mut g = g.clone();
        g.sort_unstable();
        g.dedup();
        if !gts.contains(&g) {
            gts.push(g);
        }
    }
    let k = gts.len();
    let ln_prior: Vec<f64> = (0..k)
        .map(|i| {
            if i == 0 || k == 1 {
                0.5f64.ln()
            } else {
                (0.5 / (k - 1) as f64).ln()
            }
        })
        .collect();
    let n = post.cells.len();
    let mut prob = vec![0.0f32; n * k];
    let best: Vec<(u32, f32)> = prob
        .par_chunks_exact_mut(k)
        .enumerate()
        .map(|(c, out)| {
            let score: Vec<f64> = gts
                .iter()
                .zip(&ln_prior)
                .map(|(g, lp)| {
                    lp + g
                        .iter()
                        .map(|e| {
                            let col = match e.state {
                                None => COL_LOGBF,
                                Some(PostState::Loh) => COL_LOH,
                                Some(PostState::Del | PostState::Bdel) => COL_DEL,
                                Some(_) => COL_AMP,
                            };
                            post.at(c, e.col)[col] as f64
                        })
                        .sum::<f64>()
                })
                .collect();
            let z = log_sum_exp(&score);
            let mut arg = 0;
            for (i, (o, s)) in out.iter_mut().zip(&score).enumerate() {
                *o = (s - z).exp() as f32;
                if score[i] > score[arg] {
                    arg = i;
                }
            }
            (arg as u32, 1.0 - out[0])
        })
        .collect();
    let (clone, p_tumour) = best.into_iter().unzip();
    ClonePosterior {
        genotypes: gts,
        prob,
        clone,
        p_tumour,
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segments::Segment;

    fn post_one_seg(log_bf: &[f32]) -> CellPosteriors {
        CellPosteriors {
            cells: (0..log_bf.len() as u32).collect(),
            segs: vec![0],
            log_odds: log_bf
                .iter()
                .flat_map(|&b| [b, f32::NAN, f32::NAN, b])
                .collect(),
            mu: vec![0.0; log_bf.len()],
            sigma: vec![1.0; log_bf.len()],
        }
    }

    #[test]
    fn test_clone_posterior_two_genotypes() {
        // one genotype {seg 0}: score = ln .5 + logBF against ln .5
        let post = post_one_seg(&[3.0, -3.0, 0.0]);
        let gt = vec![vec![GenotypeEvent {
            col: 0,
            state: None,
        }]];
        let cp = clone_posterior(&post, &gt);
        assert_eq!(cp.genotypes.len(), 2);
        assert_eq!(cp.clone, vec![1, 0, 0]);
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        for (c, b) in [3.0, -3.0, 0.0].iter().enumerate() {
            approx::assert_relative_eq!(cp.p_tumour[c] as f64, sig(*b), max_relative = 1e-6);
        }
    }

    #[test]
    fn test_clone_posterior_drops_duplicate_and_empty_genotypes() {
        let post = post_one_seg(&[1.0]);
        let e = GenotypeEvent {
            col: 0,
            state: None,
        };
        let cp = clone_posterior(&post, &[vec![e], vec![e, e], vec![]]);
        assert_eq!(cp.genotypes.len(), 2);
    }

    /// A consensus segment with retest priors for the posterior parity test.
    fn seg(chrom: u32, lo: u32, hi: u32, post: PostState, p: [f64; 5]) -> crate::segments::Segment {
        use crate::hmm::CnState;
        use crate::segments::Retest;
        let retest = (post != PostState::Neu).then_some(Retest {
            phi_hat: 1.0,
            s_phi: 0.1,
            llr: 50.0,
            llr_x: 50.0,
            p_neu: 0.0,
            p_loh: p[0],
            p_amp: p[1],
            p_del: p[2],
            p_bamp: p[3],
            p_bdel: p[4],
            state_post: post,
        });
        Segment {
            bulk: Some(0),
            chrom,
            seg: 1,
            lo,
            hi,
            start: lo as u64 * 1000,
            end: hi as u64 * 1000,
            n_genes: hi - lo,
            state: if post == PostState::Neu {
                CnState::Neu
            } else {
                CnState::Amp1
            },
            retest,
        }
    }

    #[test]
    fn test_cell_posteriors_match_numbat() {
        use crate::reference::tests::read_tsv;
        let genes = read_tsv("post_genes.tsv");
        let counts = read_tsv("post_counts.tsv");
        let n_genes = genes.len();
        let n_cells = counts[0].len();
        let chrom: Vec<u32> = genes.iter().map(|r| r[0].parse().unwrap()).collect();
        let lambda: Vec<f64> = genes.iter().map(|r| r[1].parse().unwrap()).collect();
        let start: Vec<u64> = (0..n_genes as u64).map(|i| i * 1000).collect();
        let end: Vec<u64> = start.iter().map(|s| s + 500).collect();
        let (mut indptr, mut indices, mut data) = (vec![0u32], Vec::new(), Vec::new());
        for c in 0..n_cells {
            for (g, row) in counts.iter().enumerate() {
                let y: u32 = row[c].parse().unwrap();
                if y > 0 {
                    indices.push(g as u32);
                    data.push(y);
                }
            }
            indptr.push(indices.len() as u32);
        }
        let csr = CsrCounts::new(&indptr, &indices, &data, n_genes).unwrap();
        let order = GenomeOrder::new(&chrom, &start, &end).unwrap();
        let profiles = Profiles {
            n_ref: 1,
            lambda: lambda.clone(),
        };
        let amp = [0.03, 0.70, 0.02, 0.25, 0.0];
        let consensus = vec![
            seg(1, 0, 400, PostState::Neu, [0.0; 5]),
            seg(1, 400, 600, PostState::Amp, amp),
            seg(2, 600, 700, PostState::Amp, amp),
            seg(2, 700, 1200, PostState::Del, [0.10, 0.04, 0.80, 0.0, 0.06]),
        ];
        let cells: Vec<u32> = (0..n_cells as u32).collect();
        let post = cell_posteriors(
            &csr,
            &order,
            &profiles,
            &cells,
            &vec![0; n_cells],
            &consensus,
            false,
        )
        .unwrap();
        assert_eq!(post.segs, vec![1, 2, 3]);
        let col = |s: &str| match s {
            "1b" => 0,
            "2a" => 1,
            _ => 2,
        };
        let (mut max_fit, mut max_lo) = (0.0f64, 0.0f64);
        for r in read_tsv("post_expected.tsv") {
            let c: usize = r[0].parse().unwrap();
            let f = |i: usize| -> f64 { r[i].to_lowercase().parse().unwrap() };
            max_fit = max_fit
                .max(((post.mu[c] as f64) - f(2)).abs())
                .max(((post.sigma[c] as f64) - f(3)).abs() / f(3));
            let got = post.at(c, col(&r[1]));
            for (k, i) in [(COL_LOGBF, 4), (COL_LOH, 5), (COL_DEL, 6), (COL_AMP, 7)] {
                let (g, w) = (got[k] as f64, f(i));
                if w == f64::NEG_INFINITY {
                    assert_eq!(g, w);
                } else {
                    max_lo = max_lo.max((g - w).abs() / w.abs().max(1.0));
                }
            }
        }
        println!("max |dmu|, rel dsigma {max_fit:.2e}; max log-odds diff {max_lo:.2e}");
        assert!(max_fit < 1e-4);
        assert!(max_lo < 1e-4);
    }

    #[test]
    fn test_cell_posteriors_overlap_first_segment_owns_genes() {
        use crate::reference::tests::read_tsv;
        let genes = read_tsv("post_genes.tsv");
        let counts = read_tsv("post_counts.tsv");
        let (n_genes, n_cells) = (genes.len(), counts[0].len());
        let chrom: Vec<u32> = genes.iter().map(|r| r[0].parse().unwrap()).collect();
        let lambda: Vec<f64> = genes.iter().map(|r| r[1].parse().unwrap()).collect();
        let start: Vec<u64> = (0..n_genes as u64).map(|i| i * 1000).collect();
        let end: Vec<u64> = start.iter().map(|s| s + 500).collect();
        let (mut indptr, mut indices, mut data) = (vec![0u32], Vec::new(), Vec::new());
        for c in 0..n_cells {
            for (g, row) in counts.iter().enumerate() {
                let y: u32 = row[c].parse().unwrap();
                if y > 0 {
                    indices.push(g as u32);
                    data.push(y);
                }
            }
            indptr.push(indices.len() as u32);
        }
        let csr = CsrCounts::new(&indptr, &indices, &data, n_genes).unwrap();
        let order = GenomeOrder::new(&chrom, &start, &end).unwrap();
        let profiles = Profiles { n_ref: 1, lambda };
        let amp = [0.03, 0.70, 0.02, 0.25, 0.0];
        let cells: Vec<u32> = (0..n_cells as u32).collect();
        let run = |second_lo: u32| {
            let consensus = vec![
                seg(1, 0, 400, PostState::Neu, [0.0; 5]),
                seg(1, 400, 600, PostState::Amp, amp),
                seg(2, 600, 700, PostState::Amp, amp),
                seg(
                    2,
                    second_lo,
                    1200,
                    PostState::Del,
                    [0.10, 0.04, 0.80, 0.0, 0.06],
                ),
            ];
            cell_posteriors(
                &csr,
                &order,
                &profiles,
                &cells,
                &vec![0; n_cells],
                &consensus,
                false,
            )
            .unwrap()
        };
        // genes 650..700 lie in both chr 2 segments; the first one owns them
        let (overlap, disjoint) = (run(650), run(700));
        assert_eq!(overlap.log_odds, disjoint.log_odds);
        assert_eq!(overlap.mu, disjoint.mu);
    }

    #[test]
    fn test_entropy_bounds() {
        let post = post_one_seg(&[0.0, 0.0]);
        approx::assert_relative_eq!(post.entropy()[0], 1.0, max_relative = 1e-6);
        let post = post_one_seg(&[40.0, -40.0]);
        assert!(post.entropy()[0] < 1e-12);
    }
}
