# Fixtures for SPEC 4: Poisson-lognormal density, l_lnpois, fit_lnpois,
# phi MLE and the Laplace posterior of phi.
#
# Run from the repo root:
#   R_LIBS=<oracle lib> Rscript reference/pln.R
#
# numbat is not installed (it pulls in scistreer). The two numbat functions
# needed are parsed out of ~/repos/others/numbat/R/utils.R and evaluated alone.

suppressPackageStartupMessages({
  library(hahmmr)
  library(poilog)
  library(tibble)
})

out_dir <- "reference/fixtures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

numbat_fns <- c("calc_phi_mle_lnpois", "approx_phi_post")
exprs <- parse("~/repos/others/numbat/R/utils.R", keep.source = FALSE)
for (e in exprs) {
  if (is.call(e) && as.character(e[[1]]) %in% c("=", "<-") &&
      as.character(e[[2]]) %in% numbat_fns) {
    eval(e, globalenv())
  }
}
stopifnot(all(vapply(numbat_fns, exists, logical(1))))

fmt <- function(v) formatC(v, digits = 17, format = "g")

## ---- density grid ----

# Log of the SPEC 4.1 density by adaptive quadrature around the mode,
# integrand shifted by its maximum so nothing underflows.
ln_pln_ref <- function(x, mu, sigma, half_width) {
  h <- function(z) z * x - exp(z) - (z - mu)^2 / (2 * sigma^2)
  dh <- function(z) x - exp(z) - (z - mu) / sigma^2
  lo <- min(mu, if (x > 0) log(x) else mu) - 50
  hi <- max(mu, if (x > 0) log(x) else mu) + 50
  m <- uniroot(dh, c(lo, hi), tol = 1e-14)$root
  hm <- h(m)
  f <- function(z) exp(h(z) - hm)
  if (is.null(half_width)) {
    # limits where the integrand has dropped by exp(-80)
    g <- function(z) h(z) - hm + 80
    a <- uniroot(g, c(m - 100 * sigma, m), tol = 1e-12)$root
    b <- uniroot(g, c(m, m + 100 * sigma), tol = 1e-12)$root
  } else {
    a <- m - half_width * sigma
    b <- m + half_width * sigma
  }
  i1 <- integrate(f, a, m, rel.tol = 1e-12, abs.tol = 0, subdivisions = 10000L)$value
  i2 <- integrate(f, m, b, rel.tol = 1e-12, abs.tol = 0, subdivisions = 10000L)$value
  hm - lgamma(x + 1) - 0.5 * log(2 * pi * sigma^2) + log(i1 + i2)
}

grid <- expand.grid(
  x = c(0, 1, 2, 5, 20, 100, 1000),
  mu = c(-8, -3, 0, 2, 6),
  sigma = c(0.01, 0.1, 0.5, 1, 2)
)
grid$ln_ref <- mapply(ln_pln_ref, grid$x, grid$mu, grid$sigma,
                      MoreArgs = list(half_width = NULL))
grid$ln_ref_wide <- mapply(ln_pln_ref, grid$x, grid$mu, grid$sigma,
                           MoreArgs = list(half_width = 40))
grid$ln_poilog <- log(mapply(poilog::dpoilog, grid$x, grid$mu, grid$sigma))
grid$ln_hahmmr <- hahmmr::dpoilog(grid$x, grid$mu, grid$sigma, log = TRUE)

cat("max |ln_ref - ln_ref_wide|:", max(abs(grid$ln_ref - grid$ln_ref_wide)), "\n")

write.table(
  data.frame(x = grid$x, mu = grid$mu, sigma = grid$sigma,
             ln_ref = fmt(grid$ln_ref), ln_ref_wide = fmt(grid$ln_ref_wide),
             ln_poilog = fmt(grid$ln_poilog), ln_hahmmr = fmt(grid$ln_hahmmr)),
  file.path(out_dir, "pln_grid.tsv"), sep = "\t", quote = FALSE, row.names = FALSE
)

# Held-out random points, so the node count is not tuned to the grid alone.
set.seed(4242)
n_rand <- 400
rand <- data.frame(
  x = c(rep(0, n_rand / 4), round(exp(runif(3 * n_rand / 4, 0, log(2000))))),
  mu = runif(n_rand, -10, 8),
  sigma = exp(runif(n_rand, log(0.01), log(2.5)))
)
rand$ln_ref <- mapply(ln_pln_ref, rand$x, rand$mu, rand$sigma,
                      MoreArgs = list(half_width = NULL))
write.table(
  data.frame(x = rand$x, mu = fmt(rand$mu), sigma = fmt(rand$sigma), ln_ref = fmt(rand$ln_ref)),
  file.path(out_dir, "pln_random.tsv"), sep = "\t", quote = FALSE, row.names = FALSE
)

## ---- gene sets ----

# Zeros dominant: depth d spread over n genes with gamma(0.3) proportions,
# lognormal noise with mean -1.
sim_set <- function(n, d, lambda, phi, mu0 = -1, sig0 = 0.8) {
  rpois(n, phi * d * lambda * exp(rnorm(n, mu0, sig0)))
}

# Tight-reference l_lnpois, unfloored (SPEC 4.1, D9).
l_ref <- function(y, lambda, d, mu, sigma, phi) {
  off <- mu + log(phi * d * lambda)
  sum(mapply(ln_pln_ref, y, off, MoreArgs = list(sigma = sigma, half_width = NULL)))
}

set.seed(20261116)
meta <- list()
fixed_mu <- -0.5
fixed_sig <- 0.7
fixed_phi <- 1.3
for (n in c(500L, 2000L, 5000L)) {
  d <- as.integer(round(1.5 * n))
  g <- rgamma(n, shape = 0.3)
  lambda <- g / sum(g)
  y_base <- sim_set(n, d, lambda, 1)
  cols <- list(lambda = fmt(lambda), y_base = y_base)
  fit <- fit_lnpois_cpp(y_base, lambda, d)
  base_id <- sprintf("n%d_base", n)
  meta[[base_id]] <- data.frame(
    set = base_id, n = n, d = d, phi_true = 1, zero_frac = mean(y_base == 0),
    fit_mu = fmt(fit[1]), fit_sigma = fmt(fit[2]),
    fit_loglik = fmt(l_lnpois(y_base, lambda, d, fit[1], fit[2])),
    l_fixed = fmt(l_lnpois(y_base, lambda, d, fixed_mu, fixed_sig, fixed_phi)),
    l_fixed_ref = fmt(l_ref(y_base, lambda, d, fixed_mu, fixed_sig, fixed_phi)),
    phi_mle = NA, phi_post_mean = NA, phi_post_sd = NA
  )
  for (phi in c(0.5, 1, 1.5)) {
    y <- sim_set(n, d, lambda, phi)
    id <- sprintf("n%d_phi%s", n, phi)
    mle <- calc_phi_mle_lnpois(y, lambda, d, fit[1], fit[2], lower = 0.1, upper = 10)
    post <- approx_phi_post(y, lambda, d, mu = fit[1], sig = fit[2])
    cols[[sprintf("y_phi%s", phi)]] <- y
    meta[[id]] <- data.frame(
      set = id, n = n, d = d, phi_true = phi, zero_frac = mean(y == 0),
      fit_mu = fmt(fit[1]), fit_sigma = fmt(fit[2]),
      fit_loglik = NA,
      l_fixed = fmt(l_lnpois(y, lambda, d, fixed_mu, fixed_sig, fixed_phi)),
      l_fixed_ref = fmt(l_ref(y, lambda, d, fixed_mu, fixed_sig, fixed_phi)),
      phi_mle = fmt(unname(mle)), phi_post_mean = fmt(post$phi_mle),
      phi_post_sd = fmt(post$phi_sigma)
    )
  }
  write.table(as.data.frame(cols), file.path(out_dir, sprintf("pln_genes_n%d.tsv", n)),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

write.table(do.call(rbind, meta), file.path(out_dir, "pln_sets_meta.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
cat("fixed params for l_fixed: mu", fixed_mu, "sigma", fixed_sig, "phi", fixed_phi, "\n")
print(do.call(rbind, meta)[, c("set", "zero_frac", "fit_mu", "fit_sigma", "phi_mle", "phi_post_mean", "phi_post_sd")])
