"""Channel calibration: snr_db means Eb/N0, so sigma must account for rate.

sigma^2 = 1 / (2 * R * Eb/N0_lin).  Proof path: uncoded BPSK hard-decision
BER must match Q(sqrt(2*Eb/N0)) regardless of rate bookkeeping.
"""
import math
import numpy as np
from polar.channel_sc import snr_to_sigma, awgn_llr


def q(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def test_sigma_calibration():
    # R=1/2, Eb/N0=0dB -> sigma^2 = 1/(2*0.5*1) = 1
    assert snr_to_sigma(0.0, rate=0.5) == 1.0
    # R=1, Eb/N0=0dB -> sigma^2 = 1/2
    assert abs(snr_to_sigma(0.0, rate=1.0) - math.sqrt(0.5)) < 1e-12


def test_uncoded_bpsk_matches_theory():
    rng = np.random.default_rng(7)
    for snr_db in (0.0, 3.0):
        x = rng.integers(0, 2, size=200_000).astype(np.uint8)
        llr = awgn_llr(x, snr_db, rng, rate=1.0)
        ber = np.mean((llr < 0).astype(np.uint8) != x)
        theory = q(math.sqrt(2 * 10 ** (snr_db / 10)))
        assert abs(ber - theory) < 0.004, (snr_db, ber, theory)
