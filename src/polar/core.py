"""Stage 0 — Polar code fundamentals (Arikan 2009)."""
import numpy as np
from math import exp, sqrt, pi

F = np.array([[1, 0], [1, 1]], dtype=np.uint8)  # Arikan kernel


def kron_power(n: int) -> np.ndarray:
    """F^{⊗n}, the n-fold Kronecker power of the Arikan kernel.

    Reference only: shows the matrix form of the transform that `encode`
    applies as a butterfly. E.g. N=4 gives rows [1000, 1100, 1010, 1111],
    so x0 = u0^u1^u2^u3 and x3 = u3.
    """
    G = np.array([[1]], dtype=np.uint8)
    for _ in range(n):
        G = np.kron(G, F) % 2
    return G.astype(np.uint8)


def bit_reverse(i: int, n: int) -> int:
    """Bit-reverse an n-bit index. Reference only: Arikan writes the
    generator as B_N F^{⊗n}, where B_N is this permutation. `encode`
    absorbs it into position ordering, so this is kept to document
    the convention, not used in the data path.
    """
    return int(format(i, f"0{n}b")[::-1], 2)


def encode(u: np.ndarray) -> np.ndarray:
    """Map info/frozen vector u to codeword x = u F^{⊗n}.

    In-place butterfly over stages s = 0..n-1 with step 2^s. The stage
    order matches the contiguous-split SC recursion in channel_sc.py:
    decoding undoes these stages in reverse, which is why encoder and
    decoder must agree on it. Runs in O(N log N) with no matrix built.

    E.g. N=4, u=[u0,u1,u2,u3] -> x=[u0^u1^u2^u3, u1^u3, u2^u3, u3].
    """
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
    """Pick K info positions for a binary erasure channel with erasure
    probability eps (Arikan 2009, Prop. 5).

    Propagates the Bhattacharyya parameter Z through the polarization
    steps (upper/worse branch Z- = 2Z - Z^2, lower/better Z+ = Z^2) and
    freezes the N-K positions with the largest Z, i.e. the least
    reliable synthetic channels. Exact for the BEC; a solid default
    elsewhere. Returns a boolean mask, True = frozen to 0.

    Stage order matters: the recursion runs from the coarsest split
    (step N/2, the decoder's root) down to the finest (step 1), matching
    the SC traversal order and the tv-mc constructor. Running it
    finest-first silently bit-reverses the reliability ranking, which
    leaves truly-awful early channels unfrozen and collapses SC/SCL
    (FER ~1 even at high SNR).
    """
    n = int(np.log2(N))
    z = np.full(N, eps)
    step = N // 2
    while step >= 1:
        nxt = np.empty(N)
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = z[i + j]
                nxt[i + j] = 2 * a - a * a
                nxt[i + j + step] = a * a
        z = nxt
        step //= 2
    frozen = np.ones(N, dtype=bool)
    frozen[np.argsort(z)[:K]] = False
    return frozen


# --- Gaussian approximation (Tal-Vardy 2013 construction) ---
def _phi(x: float) -> float:
    """Chung's phi function: E[tanh(L/2)] for L ~ N(x, 2x).

    Maps a mean LLR to [0, 1], falling from phi(0) = 1 to phi(inf) = 0.
    Needed because the reliability of the upper (check-node) branch has
    no closed form under the Gaussian approximation, so it is routed
    through phi, combined in the probability domain, and mapped back
    with _phi_inv. Piecewise Ha et al. approximation; good to ~1e-3.
    """
    if x < 1e-3:
        return 1.0
    # piece-wise approx (Chung-Richardson-Urbanke / Ha et al.)
    if x < 12:
        return exp(-0.4527 * x**0.86 + 0.0218)
    return sqrt(pi / x) * exp(-x / 4) * (1 - 10 / (7 * x))


def _phi_inv(y: float) -> float:
    """Inverse of _phi by bisection (phi is monotone decreasing).

    Inputs are clamped into [1e-12, 1-1e-12] because the recursion can
    produce exact 0/1 at deep levels, where the true inverse is
    infinite/zero. 60 bisection steps pin the result to ~1e-12.
    """
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
    """Pick K info positions for BPSK/AWGN at a given design SNR.

    Tracks the mean LLR m of each synthetic channel from m0 = 2/sigma^2
    (sigma from the design Eb/N0 at rate K/N). Upper branch goes through
    phi and back; lower branch doubles. Freezes the N-K smallest means.
    Use a design SNR near the operating point: too low wastes good
    channels, too high trusts channels that will fail. Returns a boolean
    mask, True = frozen to 0.

    As in frozen_set_bec, stages run coarsest-first (step N/2 down to 1)
    to match SC traversal order; finest-first would bit-reverse the ranking.
    """
    from .channel_sc import snr_to_sigma
    sigma = snr_to_sigma(design_snr_db, rate=K / N)
    m = np.full(N, 2.0 / sigma**2)
    step, n = N // 2, int(np.log2(N))
    for _ in range(n):
        nxt = np.empty(N)
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = m[i + j]
                nxt[i + j] = _phi_inv(1 - (1 - _phi(a)) ** 2)
                nxt[i + j + step] = 2 * a
        m = nxt
        step //= 2
    frozen = np.ones(N, dtype=bool)
    frozen[np.argsort(m)[-K:]] = False
    return frozen
