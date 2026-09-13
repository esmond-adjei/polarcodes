"""Regression coverage for list decoding (previously untested)."""
import numpy as np

from polar.core import awgn_construction, encode
from polar.channel_sc import awgn_llr, sc_decode
from polar.scl import scl_decode, ca_scl_decode, path_metric, _scl_decode_with_pm
from polar.crc import append_crc, check_crc
from polar.construction import tv_mc_construction
from polar.sim import run_point, run_ml_bound_point


def test_scl_L1_matches_sc():
    for N in (16, 64):
        frozen = awgn_construction(N, N // 2, 1.0)
        rng = np.random.default_rng(100 + N)
        u = np.zeros(N, dtype=np.uint8)
        u[np.where(~frozen)[0]] = rng.integers(0, 2, size=N // 2)
        llr = awgn_llr(encode(u), 2.0, rng, rate=0.5)
        assert np.array_equal(scl_decode(llr, frozen, 1)[0], sc_decode(llr, frozen))


def test_scl_noiseless_best_matches_tx():
    N, K = 32, 16
    frozen = awgn_construction(N, K, 1.0)
    info_idx = np.where(~frozen)[0]
    rng = np.random.default_rng(3)
    u = np.zeros(N, dtype=np.uint8)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = np.where(encode(u) == 0, 10.0, -10.0)
    assert np.array_equal(scl_decode(llr, frozen, 4)[0], u)


def test_scl_list_monotonicity():
    # Larger L can only improve (or tie) the best-path metric, and on
    # average should not hurt FER. Use a fixed seed and enough blocks for
    # a stable ordering at N=128.
    N, K = 128, 64
    frozen = awgn_construction(N, K, 1.0)
    rng = np.random.default_rng(11)
    # Best-PM monotonicity on a single noisy block.
    u = np.zeros(N, dtype=np.uint8)
    u[np.where(~frozen)[0]] = rng.integers(0, 2, size=K)
    llr = awgn_llr(encode(u), 1.5, rng, rate=0.5)
    pms = []
    for L in (1, 2, 4, 8):
        _, pm = _scl_decode_with_pm(llr, frozen, L)
        pms.append(float(pm[0]))
    for a, b in zip(pms, pms[1:]):
        assert b <= a + 1e-9


def test_scl_fer_monotonic_trend():
    N, K = 128, 64
    frozen = awgn_construction(N, K, 1.0)
    fers = []
    for L in (1, 2, 4):
        dec = "sc" if L == 1 else "scl"
        _, fer, _ = run_point(N, K, frozen, 2.0, decoder=dec, L=L,
                              n_blocks=200, seed=999)
        fers.append(fer)
    # List should not be dramatically worse than SC on average; allow noise.
    assert fers[0] >= 0.0
    assert max(fers) <= 1.0
    # At least one list size ties or beats SC (holds robustly at 2 dB).
    assert min(fers[1:]) <= fers[0] + 0.05


def test_ca_scl_crc_selection():
    N, K_payload = 64, 32
    K = K_payload + 16
    frozen = awgn_construction(N, K, 1.0)
    info_idx = np.where(~frozen)[0]
    rng = np.random.default_rng(5)
    msg = rng.integers(0, 2, size=K_payload, dtype=np.uint8)
    info = append_crc(msg)
    u = np.zeros(N, dtype=np.uint8)
    u[info_idx] = info
    llr = awgn_llr(encode(u), 3.0, rng, rate=K_payload / N)
    out = ca_scl_decode(llr, frozen, info_idx, 8)
    # At high SNR the CRC-aided choice should recover the payload.
    assert np.array_equal(out[info_idx][:K_payload], msg)
    # Fallback: garbage LLRs with L=1 still returns a full vector.
    llr_bad = np.zeros(N)
    out_bad = ca_scl_decode(llr_bad, frozen, info_idx, 1)
    assert out_bad.shape == (N,)


def test_path_metric_matches_decoder_pm():
    N, K = 32, 16
    frozen = awgn_construction(N, K, 1.0)
    rng = np.random.default_rng(9)
    u = np.zeros(N, dtype=np.uint8)
    u[np.where(~frozen)[0]] = rng.integers(0, 2, size=K)
    llr = awgn_llr(encode(u), 2.0, rng, rate=0.5)
    paths, pms = _scl_decode_with_pm(llr, frozen, 4)
    for p, pm in zip(paths, pms):
        assert abs(path_metric(llr, p) - float(pm)) < 1e-9


def test_tv_mc_deterministic_and_shaped():
    f1 = tv_mc_construction(64, 32, 1.0, samples=4096, seed=123)
    f2 = tv_mc_construction(64, 32, 1.0, samples=4096, seed=123)
    assert np.array_equal(f1, f2)
    assert f1.shape == (64,) and f1.dtype == bool
    assert int(np.sum(~f1)) == 32
    f3 = tv_mc_construction(64, 32, 1.0, samples=4096, seed=999)
    # Different seed may give a slightly different set, but must stay valid.
    assert int(np.sum(~f3)) == 32


def test_sim_harness_smoke():
    N, K = 64, 32
    frozen = awgn_construction(N, K, 1.0)
    ber, fer, n = run_point(N, K, frozen, 3.0, decoder="scl", L=2,
                            n_blocks=20, seed=0)
    assert n == 20 and 0.0 <= ber <= 1.0 and 0.0 <= fer <= 1.0
    lb, scl_fer, _ = run_ml_bound_point(N, K, frozen, 3.0, n_blocks=10,
                                        L=4, seed=0)
    assert 0.0 <= lb <= scl_fer <= 1.0


def test_bec_construction_order_decodes():
    # Locks the polarization stage order: coarsest-first, matching SC
    # traversal. Finest-first bit-reverses the ranking, leaves truly-awful
    # early channels unfrozen, and pins FER near 1 even at high SNR.
    from polar.core import frozen_set_bec
    N, K = 256, 128
    frozen = frozen_set_bec(N, K, 0.5)
    _, fer, _ = run_point(N, K, frozen, 2.5, decoder="sc", L=1,
                          n_blocks=200, seed=1)
    assert fer < 0.5, f"BEC set fails to decode: FER={fer}"


def test_ga_construction_order_decodes():
    N, K = 256, 128
    frozen = awgn_construction(N, K, 2.0)
    _, fer, _ = run_point(N, K, frozen, 2.5, decoder="sc", L=1,
                          n_blocks=200, seed=1)
    assert fer < 0.5, f"GA set fails to decode: FER={fer}"


def test_constructions_agree_with_tvmc():
    # Independent constructors for the same codec order must pick nearly
    # the same positions; the reversed-stage bug dropped this to ~72%.
    from polar.core import frozen_set_bec
    N, K = 256, 128
    tvmc = set(np.where(~tv_mc_construction(N, K, 2.0, samples=8192,
                                            seed=12345))[0])
    for frozen in (frozen_set_bec(N, K, 0.5), awgn_construction(N, K, 2.0)):
        overlap = len(set(np.where(~frozen)[0]) & tvmc) / K
        assert overlap > 0.90, f"construction overlap {overlap:.1%}"
