"""Channel-dependent polar-code construction helpers.

The paper's simulations cite Tal--Vardy's channel-construction method. This
module provides two practical constructors:

* ``ga``: Gaussian approximation, useful for fast experiments.
* ``tv_mc``: Monte-Carlo density evolution of the BMS bit channels. It avoids
  the Gaussian assumption and is intended as a faithful, reproducible
  numerical approximation to the Tal--Vardy density-evolution construction.

The exact Tal--Vardy construction represents bit channels by quantized BMS
channels and applies degrading/upgrading operations. ``tv_mc`` keeps the same
bit-channel recursion but estimates the conditional LLR distributions with
common random samples, which is much easier to audit in Python.
"""
from __future__ import annotations
import numpy as np
from .core import awgn_construction
from .channel_sc import snr_to_sigma


def _boxplus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def tv_mc_construction(N: int, K: int, design_snr_db: float = 2.0,
                       samples: int = 200_000, seed: int = 12345) -> np.ndarray:
    """Monte-Carlo density-evolution construction for BI-AWGN.

    The LLR distribution is conditioned on the all-zero transmitted bit.
    At every polarization stage:
      W^- : L = boxplus(L1, L2)
      W^+ : L = L1 + L2
    Synthetic-channel error probabilities are estimated as P(L < 0), with
    half the zero mass. The K smallest error probabilities become information
    channels.

    This is an approximation to the Tal--Vardy construction, not a claim of
    bit-for-bit identity with their quantized implementation. Increase
    ``samples`` for tighter construction reproducibility.
    """
    if N < 1 or N & (N - 1):
        raise ValueError("N must be a power of two")
    if not 0 < K <= N:
        raise ValueError("K must satisfy 0 < K <= N")
    if samples < 1024:
        raise ValueError("samples should be >= 1024 for construction; 4096+ is recommended")

    rng = np.random.default_rng(seed)
    sigma = snr_to_sigma(design_snr_db, rate=K / N)
    # Conditional channel LLR for transmitted zero: L ~ N(2/sigma^2, 4/sigma^2).
    base = 2.0 / sigma**2 + (2.0 / sigma) * rng.standard_normal(samples)
    channels = [base]
    while len(channels) < N:
        nxt = []
        for llr in channels:
            # Use independent samples from the same empirical distribution.
            a = llr
            b = llr[rng.integers(0, samples, size=samples)]
            nxt.append(_boxplus(a, b))
            nxt.append(a + b)
        channels = nxt
    pe = np.empty(N)
    for i, llr in enumerate(channels):
        pe[i] = np.mean(llr < 0.0) + 0.5 * np.mean(llr == 0.0)
    info = np.argsort(pe)[:K]
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    return frozen
