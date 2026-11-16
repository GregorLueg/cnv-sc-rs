# Fixtures for SPEC 3: smoothed signal averaged over bins (SPEC 3.2), on the
# data simulated by reference/reference.R (run that first).
#
# Cases:
#   d1_b100, d1_b37  numbat's transform, then caTools::runmean per chromosome
#                    (Deviation D1), binned with bin_size 100 and 37
#   numbat1          unmodified smooth_expression on chromosome 1 alone, where
#                    D1 makes no difference
#
# Run from the repo root:
#   R_LIBS=<oracle lib> Rscript reference/smooth.R

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
rtsv <- function(name) read.delim(file.path(out_dir, name))

genes <- rtsv("ref_genes.tsv")
gtf <- data.frame(CHROM = genes$chrom, gene_start = genes$start, gene_end = genes$end,
                  gene = paste0("g", genes$gene))
gtf <- gtf[order(gtf$CHROM, gtf$gene_start), ]
cnt <- rtsv("ref_counts.tsv")
count_mat <- sparseMatrix(i = cnt$gene + 1L, j = cnt$cell + 1L, x = as.numeric(cnt$count),
                          dims = c(nrow(genes), max(cnt$cell) + 1L),
                          dimnames = list(paste0("g", genes$gene), NULL))
lam <- rtsv("ref_lambdas.tsv")
lambdas_bar <- setNames(lam$lambda_bar, paste0("g", lam$gene))

# numbat's smooth_expression (N/R/utils.R:103-128) with the running mean per
# chromosome (Deviation D1). Everything before the runmean is verbatim.
smooth_d1 <- function(count_mat, lambdas_ref, gtf, window = 101) {
  mut_expressed <- filter_genes(count_mat, lambdas_ref, gtf)
  count_mat <- count_mat[mut_expressed, , drop = FALSE]
  lambdas_ref <- lambdas_ref[mut_expressed]
  exp_mat <- scale_counts(count_mat)
  exp_mat_norm <- log2(exp_mat * 1e6 + 1) - log2(lambdas_ref * 1e6 + 1)
  cap <- 3
  exp_mat_norm[exp_mat_norm > cap] <- cap
  exp_mat_norm[exp_mat_norm < -cap] <- -cap
  row_mean <- apply(exp_mat_norm, 2, function(x) mean(x, na.rm = TRUE))
  exp_mat_norm <- t(apply(exp_mat_norm, 1, "-", row_mean))
  chrom <- gtf$CHROM[match(mut_expressed, gtf$gene)]
  out <- exp_mat_norm
  for (ch in unique(chrom)) {
    rows <- which(chrom == ch)
    x <- exp_mat_norm[rows, , drop = FALSE]
    out[rows, ] <- if (length(rows) >= window) {
      caTools::runmean(x, k = window, align = "center")
    } else {
      # caTools errors on a matrix shorter than k; SPEC 3.1 truncated windows.
      h <- window %/% 2
      n <- length(rows)
      t(sapply(seq_len(n), function(i) colMeans(x[max(1, i - h):min(n, i + h), , drop = FALSE])))
    }
  }
  out
}

# SPEC 3.2: runs of bin_size kept genes per chromosome; a remainder below
# bin_size/2 merges into the previous bin. lo/hi are 0-based kept-gene
# indices, hi exclusive.
bin_layout <- function(chrom, bin_size) {
  do.call(rbind, lapply(unique(chrom), function(ch) {
    idx <- which(chrom == ch)
    n <- length(idx)
    lo <- seq(0, n - 1, by = bin_size)
    hi <- pmin(lo + bin_size, n)
    m <- length(lo)
    if (m > 1 && 2 * (hi[m] - lo[m]) < bin_size) {
      hi[m - 1] <- n
      lo <- lo[-m]
      hi <- hi[-m]
    }
    data.frame(chrom = ch, lo = lo + idx[1] - 1L, hi = hi + idx[1] - 1L)
  }))
}

bin_means <- function(sm, layout) {
  sapply(seq_len(nrow(layout)), function(b) {
    colMeans(sm[(layout$lo[b] + 1):layout$hi[b], , drop = FALSE])
  })
}

layouts <- list()
values <- list()
kept <- list()
add_case <- function(name, sm, chrom, bin_size, cells) {
  lay <- bin_layout(chrom, bin_size)
  bm <- bin_means(sm, lay)[cells, , drop = FALSE]
  layouts[[name]] <<- data.frame(case = name, bin = seq_len(nrow(lay)) - 1L, lay)
  values[[name]] <<- data.frame(
    case = name, cell = rep(cells - 1L, ncol(bm)), bin = rep(seq_len(ncol(bm)) - 1L, each = length(cells)),
    value = fmt(as.vector(bm))
  )
  kept[[name]] <<- data.frame(case = name, gene = as.integer(sub("g", "", rownames(sm))))
}

sm <- smooth_d1(count_mat, lambdas_bar, gtf)
chrom <- gtf$CHROM[match(rownames(sm), gtf$gene)]
n_cells <- ncol(count_mat)
add_case("d1_b100", sm, chrom, 100, seq_len(n_cells))
add_case("d1_b37", sm, chrom, 37, 1:100)

gtf1 <- gtf[gtf$CHROM == 1, ]
sm1 <- smooth_expression(count_mat[gtf1$gene, ], lambdas_bar, gtf1)
add_case("numbat1", sm1, rep(1L, nrow(sm1)), 100, seq_len(n_cells))

wtsv(do.call(rbind, layouts), "smooth_layout.tsv")
wtsv(do.call(rbind, values), "smooth_bins.tsv")
# Kept genes as 0-based caller indices, in genomic order.
wtsv(do.call(rbind, kept[c("d1_b100", "numbat1")]), "smooth_kept.tsv")
