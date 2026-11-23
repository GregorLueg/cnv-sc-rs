# SPEC: tier 0, expression-only CNV

The implementation source for tier 0. Transcribed from numbat @ `b444f3c`
(1.5.2, `~/repos/others/numbat`, MIT) and hahmmr (`~/repos/others/hahmmr`,
MIT). `N/` and `H/` prefix their paths. Code cites sections as `SPEC 4.3`.
Where we deviate from numbat on purpose, the section says so and section 11
lists every deviation.

Notation: `Y_gc` raw count, gene `g`, cell `c`. `lambda` reference
proportions (columns sum to 1). `LSE` log-sum-exp. `ln` natural log.

## 0. Conventions

- Genes are in genomic order (chromosome, then start) throughout. The
  caller's order is mapped once through a permutation (SPEC 1).
- Chromosomes are `u32` ids. The crate has no notion of autosomes, sex
  chromosomes or genome build. The caller drops chrX/Y if it wants numbat's
  behaviour (N/data-raw/prep_data.R:14-25 keeps 1..22 only).
- Loglikelihoods are exact, not up to a constant: the PLN density includes
  `lgamma(x+1)`, because the retest and posteriors compare loglikelihoods
  between segments.

## 1. Inputs and genome

- Counts: cells x genes CSR, `indptr: Vec<u32>`, `indices: Vec<u32>`, `data: Vec<u32>`.
- `GenomeOrder`: per gene `chrom: u32`, `start: u64`, `end: u64`. Built from
  caller order. It holds the genomic sort permutation and `chrom_ranges`
  (contiguous gene ranges per chromosome after sorting).
- Exclusion intervals `(chrom, start, end)`: genes overlapping any are
  dropped. numbat hard-codes MHC as `CHROM==6 & gene_start < 33480577 &
  gene_end > 28510120` (N/R/utils.R:137-167). The crate takes the intervals
  as a parameter. `EXCLUDE_HG38` (the default) and `EXCLUDE_HG19` hold MHC
  plus the six IG/TR loci (**Deviation D11**).
- Cells with zero total count are dropped (N/R/main.R:113-118).

## 2. Reference

### 2.1 Reference modes

- `Profiles`: genes x R matrix of proportions (normalised, columns sum to 1).
- `Cells`: indices of normal cells. Builds a 1-column profile:
  `lambda_g = sum_{c in ref} Y_gc / sum_{g,c in ref} Y_gc`
  (`aggregate_counts`, N/R/utils.R:62-95).
- `Groups`: normal cells with a label each. One `Cells` profile per label,
  giving genes x R as numbat's per-cell-type `lambdas_ref`.

### 2.2 `filter_genes` (N/R/utils.R:137-167)

Given `lambda_ref` (one vector) and a set of cells:
`lambda_obs_g = sum_c Y_gc / sum_{g,c} Y_gc`, computed after exclusion.
Keep `g` iff
`[(lambda_ref_g*1e6 > 2 and lambda_obs_g*1e6 > 2) or lambda_ref_g > mean(lambda_ref | != 0) or lambda_obs_g > mean(lambda_obs | != 0)] and lambda_ref_g > 0`.

### 2.3 `choose_ref_cor` (N/R/utils.R:9-46)

R = 1: every cell gets reference 0. Otherwise:
1. Genes are those with `lambda_{g,r}*1e6 > 2` for at least one `r`.
2. `x_gc = Y_gc / sum_g Y_gc`, summed over all annotated genes.
3. `rho_{c,r} = Pearson_g(ln(x_gc*1e6+1), ln(lambda_{g,r}*1e6+1))`.
4. The cell's reference is `argmax_r rho`, first on ties.
5. Zero-coverage cells get no reference and are dropped downstream.

Sparse note: `ln(x*1e6+1)` is 0 at zero counts, so the Pearson sums run over
nonzeros plus closed-form reference moments.

### 2.4 `lambdas_bar` (N/R/utils.R:314-326)

`w_r = #{c : ref(c) = r} / N`, `lambda_bar = Lambda w`.

### 2.5 `fit_ref_sse` (N/R/utils.R:409-455)

Per pseudobulk count vector `Y_g`. Vector reference: return it, `w = 1`.
1. Genes: annotated, `Y_g > 0`, `rowMeans(Lambda)_g > 2e-6`.
2. `lambda_obs_g = Y_g / sum Y` over those genes.
3. Minimise `f(w) = sum_g [ln(lambda_obs_g / (Lambda w / sum w)_g)]^2`,
   with `w_r >= 1e-6` and start `1/R`. numbat uses L-BFGS-B with a numerical
   gradient; we use bounded Levenberg-Marquardt on the residuals with the
   analytic Jacobian. `f` is scale-free in `w`, so a weight at the bound
   can reach a normalised value below `1e-6`.
4. Normalise `w <- w / sum w` and return `lambda_bar = Lambda w`.
5. Return `mse = f / n_genes`.
6. Parity: the final objective is <= R's * (1 + 1e-6).

## 3. Smoothed signal and bins

### 3.1 Transform (N/R/utils.R:103-128)

With `lambda_bar` (2.4) and the genes kept by 2.2 over all cells:
1. `x_gc = Y_gc / sum_{g kept} Y_gc`. Library size is over kept genes.
2. `E_gc = log2(x_gc*1e6 + 1) - log2(lambda_bar_g*1e6 + 1)`.
3. `E_gc <- clamp(E_gc, -3, 3)`.
4. Centre per cell: `E_gc <- E_gc - mean_g E_gc`, mean over kept genes.
5. Running mean, window `k = 101`, centred. At the edges the window is
   truncated, `[max(lo, i-h), min(hi, i+h)]` with `h = 50`. That is caTools
   `endrule="mean"`, checked: `runmean(c(1,2,3,10,20,30,100), 5)` gives
   `2.0 4.0 7.2 13.0 32.6 40.0 50.0`.
   **Deviation D1:** the window runs per chromosome. numbat runs one
   running mean across chromosome boundaries. A chromosome with fewer kept
   genes than `k` keeps the truncated-window rule above (caTools instead
   clamps `k` to `n` and errors on a matrix). For even `k` the window is
   `(k-1)/2` left and `k/2` right, as caTools.

### 3.2 Bins

**Deviation D2:** numbat clusters on the full smoothed gene matrix. We
reduce to fixed bins on the fly. A bin is a contiguous run of `bin_size` kept
genes (default 100, a parameter) within a chromosome. The last bin of a
chromosome takes the remainder; if that remainder is below `bin_size/2` it
merges into the previous bin. Bin value is the mean of the smoothed `E` over
its genes. Output is dense cells x bins, storage `T: CnvFloat`.

### 3.3 Sparse fused evaluation (implementation, not a model change)

For a zero count, `E_gc = clamp(-r_g, -3, 3) =: z_g`, which is the same for
every cell. Write `E_c = z + d_c`, where `d_c` is nonzero only at the cell's
nonzero genes: `d_gc = clamp(log2(x_gc*1e6+1) - r_g, -3, 3) - z_g`.
Centring, the running mean and bin averaging are all linear, so

`bins_c = B S z + B S d_c - mean(z + d_c)`,

with `S` the per-chromosome running mean and `B` bin averaging. Centring
commutes with `S` and `B` because both preserve constants. `B S z` is
computed once. `B S` applied to a unit vector at gene `g` is a fixed weight
vector over at most `ceil(101 / bin_size) + 1` bins, precomputed per gene.
Cost per cell is O(nnz * bins touched). A dense naive path stays in the
tests as the oracle for this one.

## 4. Poisson-lognormal emission

### 4.1 Density (H/R/distributions.R:19-40, H/src/poilog.cpp:28-134)

`PLN(x; mu, sigma) = (2 pi sigma^2)^{-1/2} Integral exp(z x - e^z - (z-mu)^2/(2 sigma^2) - lgamma(x+1)) dz`

hahmmr's C++ argument `sig` is the variance; the R wrapper passes `sigma^2`.
hahmmr integrates with QAGS (epsabs = epsrel = 1e-5) between limits found
around the mode, and floors `p` at `1e-15`.

Ours: Newton for the mode `m` of the concave `h(z) = z x - e^z -
(z-mu)^2/(2 sigma^2)`, then Gauss-Hermite with Laplace scaling
`s = (e^m + 1/sigma^2)^{-1/2}`. The node count is a named constant fixed by
measurement against a tight-tolerance reference (R `integrate`,
`rel.tol=1e-12`). Target: relative error <= 1e-6 on the fixture grid. No
floor (**Deviation D9**): our log-density never underflows, and hahmmr only
replaces an underflowed `p == 0` with `1e-15`, not every `p < 1e-15`.

Note that hahmmr and `poilog::dpoilog` differ by about 4e-7 relative at
`(x, mu, sigma) = (0, 1, 0.5)`, so parity against hahmmr cannot be tighter
than its own quadrature.

Sums over many genes with the same small count `x` (0 to 3) do not run the
quadrature per gene. At fixed `x` and `sigma`, the density and its
derivatives depend on the gene only through `t = mu + ln(phi d lambda_g)`,
so the quadrature runs at the nodes of piecewise Chebyshev panels covering
the group's `t` range, and each gene is interpolated. The interpolation
error is far below the quadrature's own (measurements in
`docs/PERFORMANCE.md`). This approximates the quadrature, not numbat, so it is
not a deviation. It applies to `fit_lnpois` (4.3) and `l_lnpois` (4.2).

### 4.2 Segment loglikelihood (H/R/distributions.R:53-57)

`l_lnpois(Y, lambda, d, mu, sigma, phi) = sum_g ln PLN(Y_g; mu + ln(phi d lambda_g), sigma)`.

### 4.3 `fit_lnpois` (H/src/mle.cpp:37-51)

`argmax_{mu, sigma} l_lnpois(Y, lambda, d, mu, sigma, 1)`, start `(0, 1)` (ours:
a method-of-moments start, which moves only the start, not the optimum),
`sigma >= 0.01`, no upper bound. numbat uses L-BFGS-B with a numerical
gradient. Ours is bounded Newton with the analytic gradient and Hessian,
taken from the same quadrature nodes. With `u = (z - mu)` under the
posterior of `z`:
`d ln p / d mu = E[u] / sigma^2`,
`d ln p / d sigma = E[u^2] / sigma^3 - 1 / sigma`.
Second derivatives come from the second moments. Parity: the objective is
>= R's - 1e-6 * |R's|.

### 4.4 `phi` MLE (N/R/utils.R:1425-1439)

`argmax_phi l_lnpois(Y, lambda, d, mu, sigma, phi)`, `phi in [0.1, 10]`,
start 1. Output only.

### 4.5 Laplace posterior of `phi` (`approx_phi_post`, N/R/utils.R:1508-1538)

`phi_hat = argmax l(phi)` over `[0.2, 10]`, start 1, and
`s_phi = sqrt(1 / H)` with `H = -d^2 l / d phi^2` at `phi_hat`. If it fails
or is non-finite, `phi_hat = 1` and `s_phi = 0`.

## 5. Expression HMM (hahmmr `run_joint_hmm_s15`, `exp_only`)

### 5.1 States (H/R/hahmmr.R:565-570, 626)

| idx | state | phi | w |
|---|---|---|---|
| 0 | neu | 1 | 1 |
| 1,2 | del_1 up/down | 2^-0.25 | 1 |
| 3,4 | del_2 up/down | 0.5 | 1e-10 |
| 5,6 | loh_1 up/down | 1 | 1 |
| 7,8 | loh_2 up/down | 1 | 1e-10 |
| 9,10 | amp_1 up/down | 2^0.25 | 1 |
| 11,12 | amp_2 up/down | 2.5 | 1e-10 |
| 13 | bamp | 2^0.25 | 1e-4 |
| 14 | bdel | 2^-0.25 | 1e-10 |

The phi values come from `logphi_min = 0.25`, a parameter. There are five
distinct phi values, so five emission columns are computed and mapped to
the 15 states. The `cnv_state` label strips `_up`/`_down`.

### 5.2 Transitions (H/R/hahmmr.R:679-728)

Expression only, `p_s = 0`, so the matrix is constant along the sequence:
- Same CN state, both unphased: `1 - t`.
- Same CN state, same phase: `1 - t`.
- Same CN state, other phase: 0.
- Different CN state: `t * w[to] / sum_{cn != from} w[cn]`, halved if `to`
  is phased.
`neu` is unphased; `bamp` and `bdel` are unphased. Default `t = 1e-5`.
Parity check: compare with hahmmr's `calc_trans_mat_s15` output, not this
prose.

### 5.3 Initial distribution (H/R/hahmmr.R:579-587)

The same construction with `t' = min(100 t, 1)` from `neu`.

### 5.4 Emission

`ln b_m(g) = ln PLN(Y_g; mu + ln(phi_m d lambda_g), sigma)`. Here `d` is the
pseudobulk library size over all annotated genes before filtering (7.2),
`(mu, sigma)` is the bulk baseline fit (6.1), and `lambda` is the bulk's
`lambda_bar` (2.5).

### 5.5 Viterbi (H/src/hmm.cpp:151-197)

Log space, no scaling, run per chromosome.
- `nu_0 = ln delta + e_0`.
- `nu_i[j] = max_k(nu_{i-1}[k] + ln A[k,j]) + e_i[j]`.
- Backtrack `z_i = argmax_k(ln A[k, z_{i+1}] + nu_i[k])`, first on ties.

Forward-backward is not needed for tier 0.

## 6. Segments

### 6.1 Bulk baseline (N/R/utils.R:538-552)

`fit_lnpois` on genes with `Y > 0` and `-8 < logFC < 8` that are diploid. If
none remain, it uses all of them. In the first pass every gene counts as
diploid. In the retest pass (6.6), diploid means `cnv_state in {neu}`, plus
`loh` when `use_loh`.

### 6.2 `annot_segs` (N/R/utils.R:951-974)

Run-length segments of `cnv_state` per chromosome. A segment's
`start`/`end` are its genes' min and max position; `n_genes` is its gene
count. Segment ids are per chromosome in order.

### 6.3 `smooth_segs` (N/R/utils.R:1006-1028)

Segments with `n_genes <= min_genes` (10) lose their state. The state is
then forward-filled and back-filled within the chromosome, followed by
`annot_segs` again. **Deviation D3:** numbat errors if a whole chromosome
ends up NA. We set that chromosome to `neu`.

### 6.4 Retest (`retest_cnv`, N/R/utils.R:691-788), expression only

**Deviation D4:** numbat skips this in `exp_only` mode. We run its
expression-only reduction (allele terms cancel) on every non-neu segment.

`R(l, u) = ln[Phi((u - phi_hat)/s) - Phi((l - phi_hat)/s)]`, computed as
`l_u + log1mexp(l_u - l_l)` with log-CDFs. It returns **1**, not 0, when
`s = 0`, as in N/R/utils.R:1814-1819.
- `L_n = R(2^-0.25, 2^0.25)`
- `L_d = R(0.1, 2^-0.25)`
- `L_a = R(2^0.25, 3)`

Prior `G = {neu .5, loh .1, del .1, amp .05 + .05, bamp .1, bdel .1}`:
- `Z_n = ln .5 + L_n`
- `Z_cnv = LSE(ln .1 + L_n, ln .1 + L_d, ln .05 + L_a, ln .05 + L_a, ln .1 + L_a, ln .1 + L_d)`
- `Z = LSE(Z_n, Z_cnv)`, `p_neu = exp(Z_n - Z)`
- Conditional on CNV: `p_loh = exp(ln .1 + L_n - Z_cnv)`, `p_amp = exp(ln .1 + L_a - Z_cnv)`,
  `p_del = exp(ln .1 + L_d - Z_cnv)`, `p_bamp = exp(ln .1 + L_a - Z_cnv)`,
  `p_bdel = exp(ln .1 + L_d - Z_cnv)`
- `LLR = Z_cnv - Z_n`. Also store `LLR_x = l(phi_hat) - l(1)`, which 6.5
  uses.
- `cnv_state_post` = argmax over `(loh, amp, del, bamp, bdel)`, first on
  ties. It becomes `neu` if `p_neu >= 0.5`.

Without alleles `loh` and `neu` are indistinguishable, and `bamp`/`bdel`
duplicate `amp`/`del`. **Deviation D5:** in tier 0, `loh` maps to `neu`,
`bamp` to `amp` and `bdel` to `del` after the argmax. Probabilities:
`p_amp += p_bamp`, `p_del += p_bdel`, then `p_bamp = p_bdel = 0`; `p_loh`
stays as computed. Without the fold `p_amp == p_bamp <= 0.5` always and 6.7
could never fire.

### 6.5 Consensus (`get_segs_consensus`, `resolve_cnvs`, `fill_neu_segs`, N/R/main.R:870-1175)

1. A segment is set to `neu` if `LLR < min_LLR` (5) or NA. Drop segments
   with `start == end`.
2. Non-neu segments across bulks: link two segments on the same chromosome
   when their gene-index intervals overlap and `ov/len_x >= 0.45` or
   `ov/len_y >= 0.45`. Take connected components (union-find). In each
   component, take per bulk the max of `LLR_x + LLR_y` over its segments
   (`LLR_y = 0` in tier 0) and keep **all segments of the one bulk** with the
   highest value.
   **Deviation D10:** the gene-index intervals are global gene indices in
   genomic order. numbat uses each bulk's own `snp_index`, which is not
   comparable across bulks with different gene filters.
3. `fill_neu`: the gaps between kept segments, within chromosome ranges,
   become `neu` segments. Name them `chrom` + letter in order.
4. The retest-labelled leftovers numbat produces with `retest = TRUE` are
   dropped in tier 0 (the second consensus call uses `retest = FALSE`
   anyway).

### 6.6 Retest on consensus (`retest_bulks`, N/R/main.R:1622-1673)

`use_loh` is true if the total bp of `neu` segments is < 1.5e8. Re-fit the
baseline on diploid genes (6.1) per bulk, run 6.4 on each consensus segment
per bulk, set `LLR` NA to 0, and set `cnv_state = neu` if `LLR < min_LLR`.

### 6.7 Multi-state segments (`test_multi_allelic`, N/R/main.R:1682-1733)

Over the leaf-clone bulks, per consensus segment, collect the distinct
`cnv_state_post` among bulks with `LLR > min_LLR` and `max p_* > 1 - 1e-4`.
With more than one state, the segment's state priors become 0.5 for each
present state and 0 otherwise.

## 7. Cell hierarchy and pseudobulks

### 7.1 Hierarchy (**Deviation D6**)

numbat runs exact Ward on all cells (N/R/main.R:697-726). We run
`bonsai_rs::tree::linkage::linkage_tree` (kNN Ward) on the bins (3.2), then
`bonsai_rs::tree::cluster::cluster(tree, k)` with `k = init_k` (3). Groups
follow numbat's `get_nodes_celltree` (N/R/utils.R:1545-1609): the root (all
cells), every internal node of the dendrogram restricted to the k cut that
spans >= 2 clusters, and the k leaf clusters. That is 2k-1 groups. Keep
groups with `size > min_cells` (50).

### 7.2 Pseudobulk (`get_exp_bulk`, `get_bulk`, N/R/utils.R:175-401)

Per group:
1. `D = sum` of counts over all annotated genes.
2. `Y_g = sum_{c in group} Y_gc`.
3. `lambda_bar` from 2.5 on `Y`.
4. Genes from 2.2 on this group.
5. `logFC = log2(Y/D) - log2 lambda_g`.
6. Keep genes with `|logFC| < 5` or `Y = 0`.
7. Drop `lambda = 0`.

## 8. Per-cell posteriors (`get_exp_post`, N/R/main.R:1183-1442)

Segments are the consensus with `cnv_state <- cnv_state_post` unless `neu`.
Each gene maps to its first overlapping segment. No gene filter here.

Per cell `c`, with reference `r(c)` from 2.3 and `lambda_g = Lambda[g, r(c)]`:
1. Genes are those with `lambda_g > 0`.
2. `D_c = sum Y_gc` over those genes.
3. `(mu_c, sigma_c) = fit_lnpois` on genes in `neu` segments, with `loh`
   segments included when `use_loh`. Here `use_loh` is numbat's
   `get_exp_post` rule (N/R/main.R:1334-1340): true if < 5% of the genes
   covered by the consensus are in `neu` segments, not the bp rule of 6.6.
   States are the effective states above. numbat also drops genes flagged by
   the allelic LOH segments (`segs_loh`); tier 0 has none.
4. Per non-neu segment `s`, `l_s(phi) = sum_{g in s} ln PLN(Y_gc; mu_c + ln(phi D_c lambda_g), sigma_c)`
   at `phi` in `{1, 0.5, 1.5, 2, 0.25}`:
   - `l11 = l20 = l(1)`
   - `l10 = l(0.5)`
   - `l21 = l(1.5)`
   - `l31 = l22 = l(2)`
   - `l00 = l(0.25)`

   All five come from one pass over the segment's genes.
5. Priors `pi_*` come from the segment (6.4, 6.7). Set any `pi < 0.05` to 0.
6. Combine:
   - `Z_amp = LSE(l21 + ln(pi_amp/4), l31 + ln(pi_amp/4))`
   - `Z_loh = l20 + ln(pi_loh/2)`
   - `Z_del = l10 + ln(pi_del/2)`
   - `Z_bamp = l22 + ln(pi_bamp/2)`
   - `Z_bdel = l00 + ln(pi_bdel/2)`
   - `Z_n = l11 + ln(1/2)`
   - `Z = LSE(all)`, `Z_cnv = LSE(all but Z_n)`
   - `p_x = exp(Z_x - Z)`, `p_cnv = exp(Z_cnv - Z)`, `logBF = Z_cnv - Z_n`

   **Deviation D7:** numbat later recomputes this without the 0.05 floor
   (`get_joint_post`, N/R/main.R:1554-1613). Tier 0 has no joint step, so
   the floored version is final.
7. Segment entropy: `mean_c H2(p_cnv)`. A segment counts as a confident CNV
   if it is non-neu, has entropy < 0.5 and has `LLR > min_LLR`.
8. Multi-state segments expand into one row per state (`expand_states`,
   N/R/main.R:1740-1780).

## 9. Tree-free clone posterior (**Deviation D8**, our design)

This uses numbat's `get_clone_post` formula (N/R/main.R:1022-1109) without
the tree.
1. Genotypes are the leaf clusters (7.1). Each leaf's genotype is its set of
   confident CNV segments (8.7) whose leaf-bulk `cnv_state_post` (6.6) is
   non-neu.
2. Leaves with identical genotypes merge, and the empty genotype (normal)
   is added if absent.
3. Prior: normal 0.5; each other genotype `0.5 / (n_GT - 1)`.
4. `l(c, k) = sum_s [I(s in GT_k) Z_cnv(c,s) + (1 - I) Z_n(c,s)]` over
   segments in any genotype.
5. `p(c, k) = softmax_k(ln prior_k + l(c, k))`.
6. `clone = argmax`, `p_tumour = 1 - p(c, normal)`, tumour if `p_tumour > 0.5`.

## 10. Pipeline (one pass, `run_cnv`)

1. Genome order, exclusion, drop empty cells (1).
2. Reference choice per cell (2.3) and `lambda_bar` (2.4).
3. Smoothed bins (3).
4. Hierarchy and groups (7.1).
5. Group bulks (7.2), then HMM (5).
6. `annot_segs` and `smooth_segs`, then retest (6.2-6.4).
7. Consensus (6.5), then retest on consensus (6.6). If everything is `neu`,
   stop and return the bins only.
8. Leaf bulks, then 6.6, then multi-state (6.7).
9. Per-cell posteriors (8).
10. Clone posterior (9).

numbat iterates using the tree (`max_iter = 2`); tier 0 runs one pass.

Cells are read through `CellBlocks`, in ascending blocks, in four passes:
reference choice per cell plus the gene sums of the kept cells (2, and 2.2's
`lambda_obs`); smoothing (3); the groups' gene sums (7.2); per-cell
posteriors (8). A `Reference::Cells` reference adds one pass before them
that sums the normal cells. Everything per cell is independent of the
block, and every sum is an exact integer, so the result does not depend on
the block size. Between passes only per-gene sums, the bins and the outputs
are held.

Defaults (N/R/main.R:68-77):

| param | default |
|---|---|
| `t` | 1e-5 |
| `min_LLR` | 5 |
| `init_k` | 3 |
| `min_cells` | 50 |
| `min_overlap` | 0.45 |
| `min_genes` | 10 |
| `max_entropy` | 0.5 |
| `logphi_min` | 0.25 |
| `window` | 101 |
| `cap` | 3 |
| `bin_size` | 100 (ours) |

## 11. Deviations

| id | what | why |
|---|---|---|
| D1 | running mean per chromosome; truncated windows also when a chromosome is shorter than `k` | numbat smooths across chromosome ends |
| D2 | bins instead of the full gene matrix | memory: 1M x 10k f32 is 40 GB |
| D3 | all-NA chromosome becomes `neu`, not an error | robustness |
| D4 | expression-only retest runs | numbat's `exp_only` marks every segment neu |
| D5 | loh/bamp/bdel folded into neu/amp/del | not identifiable without alleles |
| D6 | kNN Ward via bonsai-rs on bins | exact Ward is O(C^2) memory |
| D7 | 0.05 prior floor is final | no joint step in tier 0 |
| D8 | clone posterior over leaf genotypes | numbat needs the scistreer tree |
| D9 | no `1e-15` floor on the PLN density | log-space quadrature never underflows; hahmmr's floor only catches `p == 0` |
| D10 | consensus overlap on global gene indices | numbat's per-bulk `snp_index` differs between bulks |
| D11 | IGH, IGK, IGL, TRA/TRD, TRB, TRG loci excluded with MHC | numbat's refGene GTF has no IG genes there; with a full annotation, plasma and T cell receptor expression is called as CNV against a pooled reference (GSE148673 ATC2: 0.175 to 0.978 agreement) |
