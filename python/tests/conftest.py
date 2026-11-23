"""Synthetic counts shared by the tests.

The simulation of `benches/pipeline.rs`, smaller: log-normal gene means, PLN
noise (sigma 0.3), depth uniform in [2000, 6000). Cells are a reference
population of normals followed by three equal groups: normal, a gain over the
first 40 % of chr2, and that gain plus a loss over the middle half of chr5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from scipy import sparse

N_CHROM = 10
GENES_PER_CHROM = 400
N_REF = 300
N_PER_GROUP = 300


@dataclass(frozen=True)
class Sim:
    """Simulated data set."""

    X: sparse.csr_matrix
    chrom: np.ndarray
    start: np.ndarray
    end: np.ndarray
    reference: np.ndarray
    truth: np.ndarray


def simulate(seed: int = 11) -> Sim:
    """Simulate the data set; `truth` is 0 normal, 1 gain, 2 gain + loss."""
    rng = np.random.default_rng(seed)
    per, g = GENES_PER_CHROM, GENES_PER_CHROM * N_CHROM
    lam = rng.lognormal(0.0, 1.5, g)
    lam /= lam.sum()
    idx = np.arange(g)
    ch, j = idx // per, idx % per
    gain = (ch == 1) & (j < per * 2 // 5)
    loss = (ch == 4) & (j >= per // 4) & (j < per * 3 // 4)
    truth = np.r_[np.zeros(N_REF, int), np.repeat([0, 1, 2], N_PER_GROUP)]
    phi = np.ones((truth.size, g))
    phi[np.ix_(truth >= 1, gain)] = 1.5
    phi[np.ix_(truth == 2, loss)] = 0.5
    depth = rng.uniform(2000, 6000, truth.size)[:, None]
    rate = depth * lam * phi * np.exp(rng.normal(0.0, 0.3, phi.shape))
    X = sparse.csr_matrix(rng.poisson(rate).astype(np.int64))
    start = 1_000 + 10_000 * j
    return Sim(
        X=X,
        chrom=np.array([f"chr{c + 1}" for c in ch]),
        start=start,
        end=start + 5_000,
        reference=np.arange(N_REF),
        truth=truth,
    )


@pytest.fixture(scope="session")
def sim() -> Sim:
    return simulate()
