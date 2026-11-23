# Fixtures for SPEC 2: aggregate_counts, filter_genes, choose_ref_cor,
# get_lambdas_bar and fit_ref_sse on a simulated count matrix. Also writes the
# simulated inputs that reference/smooth.R reads back.
#
# Run from the repo root:
#   R_LIBS=<oracle lib> Rscript reference/reference.R
#
# numbat is not installed (it pulls in scistreer); its MIT R files are
# source()d from the local checkout.

suppressPackageStartupMessages({
  library(Matrix)
  library(dplyr)
  library(glue)
  library(logger)
})
source("~/repos/others/numbat/R/utils.R")
log_message <- function(...) invisible(NULL)

out_dir <- "reference/fixtures"
fmt <- function(v) formatC(v, digits = 17, format = "g")
wtsv <- function(df, name) {
  write.table(df, file.path(out_dir, name), sep = "\t", quote = FALSE,
              row.names = FALSE, na = "NA")
}

## ---- simulation: 300 cells x 2000 genes over 4 chromosomes ----

set.seed(7)
chrom_n <- c(`1` = 600L, `2` = 640L, `6` = 700L, `21` = 60L)
gtf <- do.call(rbind, lapply(names(chrom_n), function(ch) {
  n <- chrom_n[[ch]]
  st <- 1e6 + cumsum(sample(2e4:1e5, n))
  data.frame(CHROM = as.integer(ch), gene_start = st,
             gene_end = st + sample(1e3:5e4, n))
}))
G <- nrow(gtf)
gtf$gene <- paste0("g", seq_len(G))

# Caller order is a shuffle of genomic order; idx is the 0-based caller index.
caller <- sample(G)
gtf$idx <- match(seq_len(G), caller) - 1L

R <- 3
base <- rgamma(G, shape = 0.4, rate = 1)
lam <- sapply(seq_len(R), function(r) {
  l <- base * exp(rnorm(G, 0, 0.8))
  l[sample(G, 60)] <- 0
  l / sum(l)
})
rownames(lam) <- gtf$gene
colnames(lam) <- paste0("ref", seq_len(R))

n_cells <- 300
type <- rep(1:3, each = 100)
tumour <- seq_len(n_cells) > 220
cnv <- rep(1, G)
cnv[which(gtf$CHROM == 2)[100:300]] <- 2
cnv[which(gtf$CHROM == 6)[400:650]] <- 0.5
lib <- rlnorm(n_cells, log(400), 0.4)
Y <- sapply(seq_len(n_cells), function(c) {
  mu <- lam[, type[c]] * lib[c] * exp(rnorm(G, 0, 0.3))
  if (tumour[c]) mu <- mu * cnv
  rpois(G, mu)
})
rownames(Y) <- gtf$gene
colnames(Y) <- paste0("c", seq_len(n_cells))
count_mat <- as(Y, "CsparseMatrix")

# Inputs for the Rust side, in caller gene order.
wtsv(data.frame(gene = gtf$idx, chrom = gtf$CHROM, start = gtf$gene_start,
                end = gtf$gene_end)[order(gtf$idx), ], "ref_genes.tsv")
nz <- summary(count_mat)
wtsv(data.frame(cell = nz$j - 1L, gene = gtf$idx[nz$i], count = nz$x), "ref_counts.tsv")
wtsv(data.frame(gene = gtf$idx, ref1 = fmt(lam[, 1]), ref2 = fmt(lam[, 2]),
                ref3 = fmt(lam[, 3]))[order(gtf$idx), ], "ref_profiles.tsv")

## ---- choose_ref_cor and get_lambdas_bar ----

sc_refs <- choose_ref_cor(count_mat, lam, gtf)
ref_idx <- match(sc_refs[colnames(count_mat)], colnames(lam)) - 1L
wtsv(data.frame(cell = seq_len(n_cells) - 1L, ref = ref_idx), "ref_choose.tsv")

lambdas_bar <- get_lambdas_bar(lam, sc_refs, verbose = FALSE)

## ---- aggregate_counts (Cells mode): cells 0..99 as the normal reference ----

annot <- data.frame(cell = colnames(count_mat)[1:100], group = "normal")
agg <- aggregate_counts(count_mat, annot, verbose = FALSE)[, 1]

wtsv(data.frame(gene = gtf$idx, lambda_bar = fmt(lambdas_bar[gtf$gene]),
                aggregate = fmt(agg[gtf$gene]))[order(gtf$idx), ], "ref_lambdas.tsv")

## ---- filter_genes ----

keep_sets <- list(
  all = filter_genes(count_mat, lambdas_bar, gtf),
  subset = filter_genes(count_mat[, 151:300], lambdas_bar, gtf),
  aggregate = filter_genes(count_mat, agg, gtf)
)
wtsv(do.call(rbind, lapply(names(keep_sets), function(s) {
  data.frame(set = s, gene = sort(gtf$idx[match(keep_sets[[s]], gtf$gene)]))
})), "ref_filter.tsv")

## ---- fit_ref_sse ----

fits <- list(
  all = list(cells = 1:300, refs = 1:3),
  type2 = list(cells = 101:200, refs = 1:3),
  tumour = list(cells = 221:300, refs = 1:3),
  mix = list(cells = c(1:60, 201:240), refs = 1:3),
  two = list(cells = 1:200, refs = 2:3)
)
wtsv(do.call(rbind, lapply(names(fits), function(nm) {
  f <- fits[[nm]]
  y <- rowSums(count_mat[, f$cells, drop = FALSE])
  fit <- fit_ref_sse(y, lam[, f$refs, drop = FALSE], gtf)
  w <- rep(NA, R)
  w[f$refs] <- fit$w
  data.frame(case = nm, cells = paste(f$cells - 1L, collapse = ","),
             refs = paste(f$refs - 1L, collapse = ","),
             w1 = fmt(w[1]), w2 = fmt(w[2]), w3 = fmt(w[3]), mse = fmt(fit$mse))
})), "ref_fit.tsv")
