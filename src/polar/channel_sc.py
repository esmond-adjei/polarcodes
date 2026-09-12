"""Stage 1 — channel + Stage 2 — SC decoder (Arikan SC, LLR form).

Convention: x = u @ F^{⊗n} (butterfly encoder in core.py). Natural-order
SC with contiguous minus(f)/plus(g) splits and partial-sum re-encoding.
"""
import numpy as np


def snr_to_sigma(snr_db: float, rate: float = 1.0) -> float:
    """Eb/N0 (dB) -> AWGN std for unit-energy BPSK: sigma^2 = 1/(2*R*Eb/N0)."""
    return float(np.sqrt(1.0 / (2 * rate * 10 ** (snr_db / 10))))


def awgn_llr(x: np.ndarray, snr_db: float, rng: np.random.Generator,
             rate: float = 1.0) -> np.ndarray:
    s = 1 - 2 * x.astype(float)
    sigma = snr_to_sigma(snr_db, rate)
    return 2 * (s + rng.normal(0, sigma, size=s.shape)) / sigma**2


def f(a: float, b: float) -> float:
    sa, sb = (1 if a >= 0 else -1), (1 if b >= 0 else -1)
    return sa * sb * min(abs(a), abs(b))


def g(a: float, b: float, u: int) -> float:
    return b + (1 - 2 * int(u)) * a


def _partial_sums(u0: np.ndarray) -> np.ndarray:
    """Re-encode upper-half decisions (half-size butterfly) for the g-step."""
    from .core import encode as _enc
    return _enc(u0.copy())


def _decode_recursive(lc: np.ndarray, u_hat: np.ndarray, frozen: np.ndarray, pos: int):
    m = len(lc)
    if m == 1:
        u_hat[pos] = 0 if (frozen[pos] or lc[0] >= 0) else 1
        return np.array([u_hat[pos]], dtype=np.uint8)
    half = m // 2
    f_part = np.array([f(lc[j], lc[j + half]) for j in range(half)])
    u0 = _decode_recursive(f_part, u_hat, frozen, pos)
    s0 = _partial_sums(u0)
    g_part = np.array([g(lc[j], lc[j + half], s0[j]) for j in range(half)])
    u1 = _decode_recursive(g_part, u_hat, frozen, pos + half)
    return np.concatenate([u0, u1])  # decisions (NOT partial codeword)


def sc_decode(llr: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.uint8)
    _decode_recursive(llr.astype(float), u_hat, frozen, 0)
    return u_hat


def sc_llr(llr: np.ndarray, u_dec: np.ndarray, i: int) -> float:
    """LLR of bit i given past decisions u_dec[:i] (shared engine for SCL)."""
    lc = llr.astype(float)
    pos, m = 0, len(lc)
    while m > 1:
        half = m // 2
        if i < pos + half:
            lc = np.array([f(lc[j], lc[j + half]) for j in range(half)])
            m = half
        else:
            up = _partial_sums(u_dec[pos:pos + half].copy())
            lc = np.array([g(lc[j], lc[j + half], up[j]) for j in range(half)])
            pos += half
            m = half
    return float(lc[0])
