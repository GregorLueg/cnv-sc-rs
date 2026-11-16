# Fixtures for SPEC 5: 15-state transitions, initial distribution, Viterbi
# paths and an end-to-end expression-only HMM.
#
# Run from the repo root:
#   R_LIBS=<oracle lib> Rscript reference/hmm.R

suppressPackageStartupMessages({
  library(hahmmr)
})

out_dir <- "reference/fixtures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
fmt <- function(v) formatC(v, digits = 17, format = "g")
wtsv <- function(df, name) {
  write.table(df, file.path(out_dir, name), sep = "\t", quote = FALSE, row.names = FALSE)
}

# Same construction as H/R/hahmmr.R:565-587.
states <- c("neu", "del_1_up", "del_1_down", "del_2_up", "del_2_down",
            "loh_1_up", "loh_1_down", "loh_2_up", "loh_2_down",
            "amp_1_up", "amp_1_down", "amp_2_up", "amp_2_down", "bamp", "bdel")
states_cn <- stringr::str_remove(states, "_up|_down")
states_phase <- stringr::str_extract(states, "up|down")
w <- c(neu = 1, del_1 = 1, del_2 = 1e-10, loh_1 = 1, loh_2 = 1e-10,
       amp_1 = 1, amp_2 = 1e-10, bamp = 1e-4, bdel = 1e-10)

prior_of <- function(t) {
  sapply(seq_along(states), function(to) {
    hahmmr:::get_trans_probs_s15(t = min(t * 100, 1), p_s = 0, w,
      cn_from = "neu", phase_from = NA, cn_to = states_cn[to], phase_to = states_phase[to])
  })
}

## ---- transitions ----

trans <- list()
for (t in c(1e-5, 1e-3)) {
  A <- hahmmr:::calc_trans_mat_s15(t, 0, w, states_cn, states_phase)[, , 1]
  la <- log(A)
  delta <- prior_of(t)
  trans[[length(trans) + 1]] <- data.frame(
    t = t, from = rep(0:14, each = 15), to = rep(0:14, times = 15),
    log_a = fmt(as.vector(t(la))), log_delta = fmt(rep(log(delta), times = 15))
  )
}
wtsv(do.call(rbind, trans), "hmm_trans.tsv")

## ---- Viterbi on fixed log-emissions ----

viterbi_r <- function(log_delta, logprob, la) {
  n <- nrow(logprob)
  m <- ncol(logprob)
  logPi <- array(rep(la, n), dim = c(m, m, n))
  nu <- matrix(0, n, m)
  z <- numeric(n)
  hahmmr:::viterbi_compute(log_delta, logprob, logPi, n, m, nu, z)
}

set.seed(20261116)
la <- log(hahmmr:::calc_trans_mat_s15(1e-5, 0, w, states_cn, states_phase)[, , 1])
ld <- log(prior_of(1e-5))
cases <- list(
  list(n = 50, sd = 10, planted = FALSE),
  list(n = 300, sd = 6, planted = FALSE),
  list(n = 2000, sd = 2, planted = TRUE)
)
for (k in seq_along(cases)) {
  cs <- cases[[k]]
  e <- matrix(rnorm(cs$n * 15, -5, cs$sd), cs$n, 15)
  if (cs$planted) {
    # blocks that favour one state, so the path has real switches
    blk <- sample(c(1, 2, 10, 14, 4), 10, replace = TRUE)
    for (b in seq_along(blk)) {
      rows <- ((b - 1) * cs$n / 10 + 1):(b * cs$n / 10)
      e[rows, blk[b]] <- e[rows, blk[b]] + 6
    }
  }
  z <- viterbi_r(ld, e, la)
  cat(sprintf("viterbi case %d: n %d, switches %d, states %s\n", k, cs$n,
              sum(diff(z) != 0), paste(sort(unique(z)), collapse = ",")))
  df <- as.data.frame(matrix(fmt(e), cs$n, 15))
  df$z <- as.integer(z - 1)
  wtsv(df, sprintf("hmm_viterbi_%d.tsv", k))
}

## ---- end to end: run_joint_hmm_s15(exp_only = TRUE) ----

set.seed(7)
n_chr <- c(600, 400)
mu <- 0.1
sig <- 0.3
d <- 2e6
rows <- list()
for (ci in seq_along(n_chr)) {
  n <- n_chr[ci]
  lambda <- rgamma(n, shape = 0.6, rate = 1)
  lambda <- lambda / sum(lambda) * 0.1
  phi <- rep(1, n)
  if (ci == 1) phi[201:350] <- 1.5
  if (ci == 2) phi[101:250] <- 0.5
  y <- rpois(n, exp(mu + rnorm(n, 0, sig)) * phi * d * lambda)
  s <- run_joint_hmm_s15(
    pAD = rep(NA, n), DP = rep(NA, n), p_s = rep(0, n),
    Y_obs = y, lambda_ref = lambda, d_total = d, mu = mu, sig = sig,
    t = 1e-5, phi_amp = 2^0.25, phi_del = 2^-0.25, exp_only = TRUE
  )
  idx <- match(s, states) - 1
  cat(sprintf("e2e chrom %d: %s\n", ci, paste(names(table(s)), table(s), collapse = " ")))
  rows[[ci]] <- data.frame(chrom = ci, y = y, lambda = fmt(lambda), state = idx)
}
wtsv(do.call(rbind, rows), "hmm_e2e.tsv")
wtsv(data.frame(mu = fmt(mu), sigma = fmt(sig), d = fmt(d), t = 1e-5, logphi_min = 0.25),
     "hmm_e2e_meta.tsv")
