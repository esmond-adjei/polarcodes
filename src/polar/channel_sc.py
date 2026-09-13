"""AWGN channel and successive-cancellation primitives for polar codes."""
import numpy as np


def snr_to_sigma(snr_db: float, rate: float = 1.0) -> float:
    """Eb/N0 (dB) -> AWGN standard deviation for unit-energy BPSK."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    return float(np.sqrt(1.0 / (2.0 * rate * 10.0 ** (snr_db / 10.0))))


def awgn_llr(x: np.ndarray, snr_db: float, rng: np.random.Generator,
             rate: float = 1.0) -> np.ndarray:
    """Transmit BPSK over BI-AWGN and return exact channel LLRs."""
    s = 1.0 - 2.0 * x.astype(float)
    sigma = snr_to_sigma(snr_db, rate)
    y = s + rng.normal(0.0, sigma, size=s.shape)
    return 2.0 * y / (sigma * sigma)


def f(a: float, b: float) -> float:
    """Exact LLR-domain polar check-node (box-plus) operation.

    This is numerically stable and does not use the min-sum approximation.
    Equivalent form:
        log((1 + exp(a+b)) / (exp(a) + exp(b))).
    """
    return float(np.logaddexp(0.0, a + b) - np.logaddexp(a, b))


def g(a: float, b: float, u: int) -> float:
    """Exact variable-node update."""
    return float(b + (1 - 2 * int(u)) * a)


def _partial_sums(u0: np.ndarray) -> np.ndarray:
    from .core import encode as _enc
    return _enc(u0.copy())


def _decode_recursive(lc: np.ndarray, u_hat: np.ndarray, frozen: np.ndarray, pos: int):
    m = len(lc)
    if m == 1:
        u_hat[pos] = 0 if (frozen[pos] or lc[0] >= 0) else 1
        return np.array([u_hat[pos]], dtype=np.uint8)
    half = m // 2
    f_part = np.fromiter((f(lc[j], lc[j + half]) for j in range(half)), dtype=float, count=half)
    u0 = _decode_recursive(f_part, u_hat, frozen, pos)
    s0 = _partial_sums(u0)
    g_part = np.fromiter((g(lc[j], lc[j + half], s0[j]) for j in range(half)), dtype=float, count=half)
    u1 = _decode_recursive(g_part, u_hat, frozen, pos + half)
    return np.concatenate([u0, u1])


def sc_decode(llr: np.ndarray, frozen: np.ndarray) -> np.ndarray:
    """Successive-cancellation decode."""
    u_hat = np.zeros(len(llr), dtype=np.uint8)
    _decode_recursive(llr.astype(float), u_hat, frozen, 0)
    return u_hat


def sc_llr(llr: np.ndarray, u_dec: np.ndarray, i: int) -> float:
    """LLR of bit i conditioned on the already-decided prefix."""
    lc = llr.astype(float)
    pos, m = 0, len(lc)
    while m > 1:
        half = m // 2
        if i < pos + half:
            lc = np.fromiter((f(lc[j], lc[j + half]) for j in range(half)),
                             dtype=float, count=half)
            m = half
        else:
            up = _partial_sums(u_dec[pos:pos + half].copy())
            lc = np.fromiter((g(lc[j], lc[j + half], up[j]) for j in range(half)),
                             dtype=float, count=half)
            pos += half
            m = half
    return float(lc[0])
