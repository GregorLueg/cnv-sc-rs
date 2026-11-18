# Fixtures for SPEC 8: per-cell get_exp_likelihoods + compute_posterior.
#
# Run from the repo root:
#   R_LIBS=<oracle lib> Rscript reference/posterior.R
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

out_dir <- "reference/fixtures"
fmt <- function(v) formatC(v, digits = 17, format = "g")
wtsv <- function(df, name) {
  write.table(df, file.path(out_dir, name), sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
}

set.seed(5)
n_genes <- 1200
lambda <- rlnorm(n_genes, 0, 1.5)
lambda[c(17, 450, 901)] <- 0  # zero-lambda genes are dropped per cell
lambda <- lambda / sum(lambda)
chrom <- rep(1:2, each = 600)
# segments: genes 1..400 neu, 401..700 amp, 701..1200 del (1-based)
seg <- c(rep("1a", 400), rep("1b", 200), rep("2a", 100), rep("2b", 500))
cnv_state <- c(rep("neu", 400), rep("amp", 300), rep("del", 500))
phi_true <- function(cell) {
  p <- rep(1, n_genes)
  if (cell %% 2 == 0) p[401:700] <- 1.5
  if (cell %% 3 == 0) p[701:1200] <- 0.5
  p
}
n_cells <- 6
Y <- sapply(1:n_cells, function(c) {
  d <- 2000 + 700 * c
  rpois(n_genes, d * lambda * phi_true(c) * exp(rnorm(n_genes, 0, 0.4)))
})

priors <- data.frame(
  seg = c("1b", "2a", "2b"),
  prior_loh = c(0.03, 0.03, 0.10),
  prior_amp = c(0.70, 0.70, 0.04),
  prior_del = c(0.02, 0.02, 0.80),
  prior_bamp = c(0.25, 0.25, 0.00),
  prior_bdel = c(0.00, 0.00, 0.06)
) %>%
  # main.R:1400-1403
  mutate(across(starts_with("prior"), function(x) ifelse(x < 0.05, 0, x)))

res <- do.call(rbind, lapply(1:n_cells, function(c) {
  exp_counts <- data.frame(
    gene = seq_len(n_genes), Y_obs = Y[, c], lambda_ref = lambda,
    CHROM = chrom, seg = seg, cnv_state = cnv_state, loh = FALSE
  )
  get_exp_likelihoods(exp_counts) %>%
    left_join(priors, by = "seg") %>%
    compute_posterior() %>%
    mutate(cell = c - 1L)
}))

wtsv(data.frame(chrom = chrom, lambda = fmt(lambda),
                seg = c(rep(0, 400), rep(1, 200), rep(2, 100), rep(3, 500))),
     "post_genes.tsv")
wtsv(as.data.frame(Y), "post_counts.tsv")
wtsv(data.frame(
  cell = res$cell, seg = res$seg,
  mu = fmt(res$mu), sigma = fmt(res$sigma),
  logBF = fmt(res$logBF),
  loh = fmt(res$Z_loh - res$Z_n), del = fmt(res$Z_del - res$Z_n),
  amp = fmt(res$Z_amp - res$Z_n)
), "post_expected.tsv")
cat("wrote", nrow(res), "rows\n")
