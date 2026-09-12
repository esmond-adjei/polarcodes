"""Stage 0 — Polar code fundamentals (Arikan 2009)."""
import numpy as np
from math import exp, log, sqrt, pi

F = np.array([[1, 0], [1, 1]], dtype=np.uint8)


def kron_power(n: int) -> np.ndarray:
    G = np.array([[1]], dtype=np.uint8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    return G.astype(np.uint8)


def bit_reverse(i: int, n: int) -> int:
    return int(format(i, f"0{n}b")[::-1], 2)


def encode(u: np.ndarray) -> np.ndarray:
    """Standard non-systematic polar encoder (matches contiguous-split SC).

    In-place butterfly over stages s=0..n-1 with step 2^s. This is the
    conventional x = u·F^{⊗n} stage ordering consistent with the
    natural-order SC recursion (f on contiguous halves) in channel_sc.py.
    """
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N
    v = u.copy()
    step = 1
    for _ in range(n):
        for i in range(0, N, 2 * step):
            for j in range(step):
                v[i + j] ^= v[i + j + step]
        step *= 2
    return v


def frozen_set_bec(N: int, K: int, eps: float = 0.5) -> np.ndarray:
    """BEC Bhattacharyya recursion -> boolean frozen mask."""
    n = int(np.log2(N))
    z = np.full(N, eps)
    step = 1
    for _ in range(n):
        nxt = np.empty(N)
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = z[i + j]
                nxt[i + j] = 2 * a - a * a
                nxt[i + j + step] = a * a
        z = nxt
        step *= 2
    frozen = np.ones(N, dtype=bool)
    frozen[np.argsort(z)[:K]] = False
    return frozen


# --- Gaussian approximation (Tal-Vardy 2013 construction) ---
def _phi(x: float) -> float:
    if x < 1e-3:
        return 1.0
    # piece-wise approx (Chung-Richardson-Urbanke / Ha et al.)
    if x < 12:
        return exp(-0.4527 * x**0.86 + 0.0218)
    return sqrt(pi / x) * exp(-x / 4) * (1 - 10 / (7 * x))


def _phi_inv(y: float) -> float:
    y = min(max(y, 1e-12), 1 - 1e-12)
    # bisection on monotone decreasing phi
    lo, hi = 1e-4, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _phi(mid) > y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def awgn_construction(N: int, K: int, design_snr_db: float = 0.0) -> np.ndarray:
    """Rank synthetic channels by GA mean LLR; freeze N-K worst."""
    sigma = 10 ** (-design_snr_db / 20)  # BPSK, unit energy: sigma from Eb/N0-ish design SNR
    m = np.full(N, 2.0 / sigma**2)
    step, n = 1, int(np.log2(N))
    for _ in range(n):
        nxt = np.empty(N)
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = m[i + j]
                nxt[i + j] = _phi_inv(1 - (1 - _phi(a)) ** 2)
                nxt[i + j + step] = 2 * a
        m = nxt
        step *= 2
    frozen = np.ones(N, dtype=bool)
    frozen[np.argsort(m)[-K:]] = False
    return frozen
