# Fixtures for SPEC 4.3 edge cases of fit_lnpois: near-Poisson (sigma at its
# bound), very sparse, and underdispersed gene sets, against hahmmr.
#
# Run from the repo root:
#   R_LIBS=reference/.rlib Rscript reference/pln_fit_edge.R

suppressPackageStartupMessages(library(hahmmr))

out_dir <- "reference/fixtures"
fmt <- function(v) formatC(v, digits = 17, format = "g")

set.seed(20261117)
n <- 300L
g <- rgamma(n, shape = 0.5)
lambda <- g / sum(g)
sets <- list(
  poisson = rpois(n, 2000 * lambda),
  near_poisson = rpois(n, 2000 * lambda * exp(rnorm(n, 0, 0.05))),
  sparse = c(1L, rep(0L, n - 1L)),
  low_depth = rpois(n, 50 * lambda * exp(rnorm(n, 0, 0.5))),
  under = as.integer(round(2000 * lambda))
)
d_of <- c(poisson = 2000, near_poisson = 2000, sparse = 300, low_depth = 50, under = 2000)

rows <- lapply(names(sets), function(k) {
  y <- sets[[k]]
  d <- d_of[[k]]
  fit <- fit_lnpois_cpp(y, lambda, d)
  data.frame(set = k, d = d, fit_mu = fmt(fit[1]), fit_sigma = fmt(fit[2]),
             fit_loglik = fmt(l_lnpois(y, lambda, d, fit[1], fit[2])),
             y = paste(y, collapse = ","))
})
write.table(do.call(rbind, rows), file.path(out_dir, "pln_fit_edge.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(lambda = fmt(lambda)), file.path(out_dir, "pln_fit_edge_lambda.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
cat("wrote", length(rows), "sets\n")
