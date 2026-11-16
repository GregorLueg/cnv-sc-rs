# Fixtures for SPEC 6: annot_segs, smooth_segs, the expression-only retest
# maths, consensus (resolve_cnvs + fill_neu_segs) and test_multi_allelic.
#
# Run from the repo root:
#   R_LIBS=<oracle lib> Rscript reference/segments.R
#
# numbat is not installed (it pulls in scistreer); its MIT R files are
# source()d from the local checkout.

suppressPackageStartupMessages({
  library(hahmmr)
  library(dplyr)
  library(glue)
  library(logger)
  library(stringr)
})
source("~/repos/others/numbat/R/utils.R")
source("~/repos/others/numbat/R/main.R")
log_message <- function(...) invisible(NULL)

out_dir <- "reference/fixtures"
fmt <- function(v) formatC(v, digits = 17, format = "g")
wtsv <- function(df, name) {
  write.table(df, file.path(out_dir, name), sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
}

## ---- annot_segs and smooth_segs ----

set.seed(11)
runs <- list(
  `1` = list(c("neu", 20), c("del_1", 5), c("neu", 10), c("amp_1", 15), c("neu", 11), c("loh_1", 3)),
  `2` = list(c("amp_1", 4), c("del_1", 30), c("neu", 11), c("bamp", 12))
)
genes <- do.call(rbind, lapply(names(runs), function(ch) {
  st <- unlist(lapply(runs[[ch]], function(r) rep(r[1], as.integer(r[2]))))
  data.frame(CHROM = as.integer(ch), cnv_state = st,
             POS = cumsum(sample(1000:90000, length(st))))
}))
genes$snp_index <- seq_len(nrow(genes)) - 1L
genes$gene <- paste0("g", genes$snp_index)
genes$pAD <- NA

a1 <- annot_segs(genes)
sm <- annot_segs(smooth_segs(a1, min_genes = 10))
wtsv(data.frame(
  chrom = genes$CHROM, pos = genes$POS, gene = genes$snp_index, state = genes$cnv_state,
  seg1 = sub("^[0-9]+", "", a1$seg), seg_start1 = a1$seg_start, seg_end1 = a1$seg_end,
  n_genes1 = a1$n_genes,
  state2 = sm$cnv_state, seg2 = sub("^[0-9]+", "", sm$seg), seg_start2 = sm$seg_start,
  seg_end2 = sm$seg_end, n_genes2 = sm$n_genes
), "segs_annot.tsv")

## ---- expression-only retest maths (SPEC 6.4, alleles absent) ----

# The allele terms L_y_* are dropped (they would be a common constant), the
# rest is the summarise() of retest_cnv (N/R/utils.R:740-772) verbatim.
G <- c("11" = 0.5, "20" = 0.1, "10" = 0.1, "21" = 0.05, "31" = 0.05, "22" = 0.1, "00" = 0.1)
logphi_min <- 0.25
retest_row <- function(phi_mle, phi_sigma) {
  L_x_n <- pnorm.range.log(2^(-logphi_min), 2^logphi_min, phi_mle, phi_sigma)
  L_x_d <- pnorm.range.log(0.1, 2^(-logphi_min), phi_mle, phi_sigma)
  L_x_a <- pnorm.range.log(2^logphi_min, 3, phi_mle, phi_sigma)
  Z_cnv <- logSumExp(c(log(G)["20"] + L_x_n, log(G)["10"] + L_x_d, log(G)["21"] + L_x_a,
                       log(G)["31"] + L_x_a, log(G)["22"] + L_x_a, log(G)["00"] + L_x_d))
  Z_n <- log(G)["11"] + L_x_n
  Z <- logSumExp(c(Z_n, Z_cnv))
  p <- unname(c(
    p_neu = exp(Z_n - Z),
    p_loh = exp(log(G)["20"] + L_x_n - Z_cnv),
    p_amp = exp(log(G["31"] + G["21"]) + L_x_a - Z_cnv),
    p_del = exp(log(G)["10"] + L_x_d - Z_cnv),
    p_bamp = exp(log(G)["22"] + L_x_a - Z_cnv),
    p_bdel = exp(log(G)["00"] + L_x_d - Z_cnv)
  ))
  p[is.na(p)] <- 0
  post <- c("loh", "amp", "del", "bamp", "bdel")[which.max(p[2:6])]
  if (p[1] >= 0.5) post <- "neu"
  data.frame(phi = fmt(phi_mle), s = fmt(phi_sigma), llr = fmt(unname(Z_cnv - Z_n)),
             t(fmt(unname(p))), post = post)
}
grid <- expand.grid(phi = c(0.2, 0.5, 0.8, 1, 1.1, 1.25, 2, 5, 10),
                    s = c(0, 0.001, 0.05, 0.2, 1, 5))
rt <- do.call(rbind, mapply(retest_row, grid$phi, grid$s, SIMPLIFY = FALSE))
names(rt)[4:9] <- c("p_neu", "p_loh", "p_amp", "p_del", "p_bamp", "p_bdel")
wtsv(rt, "segs_retest_grid.tsv")

# One simulated segment end to end: baseline from neutral genes, then the
# numbat pieces on the segment.
set.seed(5)
n <- 400
lambda <- rgamma(n, 0.6, 1); lambda <- lambda / sum(lambda) * 0.2
d <- 1e6
phi <- c(rep(1, 250), rep(1.4, 150))
y <- rpois(n, exp(0.05 + rnorm(n, 0, 0.4)) * phi * d * lambda)
fit <- fit_lnpois_cpp(y[1:250], lambda[1:250], d)
seg <- 251:400
post <- approx_phi_post(y[seg], lambda[seg], d, mu = fit[1], sig = fit[2])
llr_x <- calc_exp_LLR(y[seg], lambda[seg], d, post$phi_mle, mu = fit[1], sig = fit[2])
one <- retest_row(post$phi_mle, post$phi_sigma)
wtsv(data.frame(y = y, lambda = fmt(lambda), in_seg = seq_len(n) %in% seg), "segs_retest_genes.tsv")
wtsv(cbind(data.frame(d = d, mu = fmt(fit[1]), sigma = fmt(fit[2]), llr_x = fmt(llr_x)), one),
     "segs_retest_one.tsv")

## ---- consensus: get_segs_consensus(retest = FALSE) ----

set.seed(3)
n_chr <- c(`1` = 200, `2` = 150, `3` = 160)
gpos <- do.call(rbind, lapply(names(n_chr), function(ch) {
  data.frame(CHROM = as.integer(ch), POS = cumsum(sample(1000:50000, n_chr[[ch]])))
}))
gpos$gene_index <- seq_len(nrow(gpos)) - 1L
# per sample: CHROM, first and last gene (1-based within chrom), state, LLR, LLR_x
segdef <- read.table(header = TRUE, text = "
sample CHROM a b cnv_state LLR LLR_x
1 1 1 50 neu NA NA
1 1 51 90 amp_1 30 25
1 1 91 120 amp_2 15 12
1 1 121 200 neu NA NA
1 2 1 60 del_1 3 2
1 2 61 150 neu NA NA
1 3 1 9 neu NA NA
1 3 10 50 del_1 12 9
1 3 51 160 neu NA NA
2 1 1 40 neu NA NA
2 1 41 110 amp_1 40 20
2 1 111 200 neu NA NA
2 2 1 70 del_1 50 45
2 2 71 150 neu NA NA
2 3 1 44 neu NA NA
2 3 45 120 amp_1 8 6
2 3 121 160 neu NA NA
3 1 1 55 neu NA NA
3 1 56 80 del_1 10 10
3 1 81 200 neu NA NA
3 2 1 30 neu NA NA
3 2 31 150 amp_1 20 30
3 3 1 99 neu NA NA
3 3 100 100 del_1 9 9
3 3 101 160 neu NA NA
")
bulks <- do.call(rbind, lapply(seq_len(nrow(segdef)), function(i) {
  s <- segdef[i, ]
  g <- gpos[gpos$CHROM == s$CHROM, ][s$a:s$b, ]
  data.frame(sample = s$sample, CHROM = s$CHROM, POS = g$POS, snp_index = g$gene_index,
             seg = paste0(s$CHROM, "_", i), cnv_state = s$cnv_state, cnv_state_post = s$cnv_state,
             LLR = s$LLR, LLR_x = s$LLR_x, LLR_y = 0, n_genes = nrow(g))
}))
bulks <- bulks %>% group_by(seg, sample) %>%
  mutate(seg_start_index = min(snp_index), seg_end_index = max(snp_index)) %>% ungroup()

segs_in <- bulks %>% group_by(sample, CHROM, seg, cnv_state, LLR, LLR_x, n_genes) %>%
  summarise(lo = min(snp_index), hi = max(snp_index) + 1L, seg_start = min(POS),
            seg_end = max(POS), .groups = "drop") %>%
  arrange(sample, CHROM, lo)
wtsv(segs_in %>% select(sample, CHROM, lo, hi, seg_start, seg_end, n_genes, cnv_state, LLR, LLR_x),
     "segs_cons_in.tsv")
wtsv(gpos, "segs_cons_genes.tsv")

cons <- get_segs_consensus(bulks, min_LLR = 5, min_overlap = 0.45, retest = FALSE)
cons <- as.data.frame(cons)
print(cons[, c("sample", "CHROM", "seg_start", "seg_end", "cnv_state", "seg_cons")])
wtsv(data.frame(sample = cons$sample, chrom = as.integer(as.character(cons$CHROM)),
                seg_start = cons$seg_start, seg_end = cons$seg_end, cnv_state = cons$cnv_state,
                seg = sub("^[0-9]+", "", cons$seg_cons)), "segs_cons_out.tsv")

# all neutral: every LLR below min_LLR
bulks_neu <- bulks %>% mutate(LLR = ifelse(is.na(LLR), NA, 1))
cn <- as.data.frame(get_segs_consensus(bulks_neu, min_LLR = 5, min_overlap = 0.45, retest = FALSE))
wtsv(data.frame(chrom = as.integer(as.character(cn$CHROM)), seg_start = cn$seg_start,
                seg_end = cn$seg_end, seg = sub("^[0-9]+", "", cn$seg)), "segs_cons_allneu.tsv")

## ---- test_multi_allelic ----

mbulks <- read.table(header = TRUE, text = "
sample CHROM seg_cons LLR p_amp p_del p_bdel p_loh p_bamp cnv_state_post
1 1 1b 30 0.99995 0 0 0 0 amp
2 1 1b 20 0 0.999999 0 0 0 del
3 1 1b 3 0.99999 0 0 0 0 amp
1 2 2a 40 0 0.9999999 0 0 0 del
2 2 2a 25 0 0.99999 0 0 0 del
3 2 2a 10 0.9 0 0 0 0 amp
1 3 3b 12 0.99999 0 0 0 0 amp
2 3 3b 12 0 0.99 0 0 0 del
3 3 3b 50 0 0 0 0.99999 0 loh
")
mcons <- read.table(header = TRUE, text = "
CHROM seg_cons cnv_state cnv_state_post p_amp p_del p_bdel p_loh p_bamp
1 1a neu neu NA NA NA NA NA
1 1b amp_1 amp 0.8 0.1 0.05 0.03 0.02
2 2a del_1 del 0.1 0.85 0.02 0.02 0.01
3 3a neu neu NA NA NA NA NA
3 3b amp_1 amp 0.7 0.2 0.04 0.03 0.03
")
mo <- test_multi_allelic(mbulks, mcons, min_LLR = 5, p_min = 1 - 1e-4)
wtsv(data.frame(seg_cons = mo$seg_cons, n_states = mo$n_states, cnv_states = mo$cnv_states,
                p_amp = mo$p_amp, p_del = mo$p_del, p_bdel = mo$p_bdel, p_loh = mo$p_loh,
                p_bamp = mo$p_bamp), "segs_multi_out.tsv")
wtsv(mbulks, "segs_multi_bulks.tsv")
wtsv(mcons, "segs_multi_cons.tsv")
print(mo)
