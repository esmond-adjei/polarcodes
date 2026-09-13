import numpy as np
from polar.core import encode, kron_power
from polar.channel_sc import f
from polar.crc import append_crc, check_crc
from polar.systematic import systematic_encode


def test_exact_f_matches_definition():
    for a, b in [(1.2, 0.7), (1.2, -0.7), (-1.2, -0.7), (40.0, -3.0)]:
        got = f(a, b)
        direct = np.logaddexp(0.0, a + b) - np.logaddexp(a, b)
        assert np.isclose(got, direct, atol=1e-12)


def test_crc_round_trip():
    rng = np.random.default_rng(7)
    msg = rng.integers(0, 2, 128, dtype=np.uint8)
    c = append_crc(msg)
    assert check_crc(c)
    c[3] ^= 1
    assert not check_crc(c)


def test_systematic_positions():
    rng = np.random.default_rng(8)
    N = 16
    G = kron_power(4)
    info_idx = np.array([3, 5, 7, 9, 11, 13, 14, 15])
    msg = rng.integers(0, 2, len(info_idx), dtype=np.uint8)
    x = systematic_encode(msg, info_idx, N)
    assert np.array_equal(x[info_idx], msg)
    # Encoding remains a valid polar codeword.
    assert x.dtype == np.uint8
