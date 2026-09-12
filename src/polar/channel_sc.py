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
    """Send codeword x through BPSK/AWGN and return channel LLRs.

    Bit 0 -> +1, bit 1 -> -1, plus Gaussian noise of std sigma. The LLR
    L = 2y/sigma^2 is exact for this channel (not an approximation), so
    decoder errors downstream can never be blamed on this function.
    E.g. x=[0,1] at high SNR gives [+big, -big].
    """
    s = 1 - 2 * x.astype(float)
    sigma = snr_to_sigma(snr_db, rate)
    return 2 * (s + rng.normal(0, sigma, size=s.shape)) / sigma**2


def f(a: float, b: float) -> float:
    """Check-node (upper branch) update, min-sum form.

    Answers: given LLRs a, b for two bits, what is the LLR of their XOR?
    Exact boxplus is sign(a)sign(b)*min(|a|,|b|) plus a log correction;
    dropping the correction costs almost nothing and avoids tanh/exp per
    node. Zeros map to sign +1, which only matters for exact ties.
    """
    sa, sb = (1 if a >= 0 else -1), (1 if b >= 0 else -1)
    return sa * sb * min(abs(a), abs(b))


def g(a: float, b: float, u: int) -> float:
    """Variable-node (lower branch) update.

    Answers: given LLRs a, b where the upper bit turned out to be u, what
    is the LLR of the lower bit? If u = 0 the two observations agree and
    add (b + a); if u = 1 they disagree and subtract (b - a). int(u)
    guards against numpy uint8 wraparound in 1 - 2*u.
    """
    return b + (1 - 2 * int(u)) * a


def _partial_sums(u0: np.ndarray) -> np.ndarray:
    """Re-encode upper-half decisions through a half-size butterfly.

    The g-step needs these partial sums, not the raw decisions: after
    folding the factor graph, lower-branch LLRs are conditioned on
    codeword bits of the upper segment, which are the re-encoded
    decisions. Skipping this (using raw decisions) decodes N=4 fine and
    fails silently above it. The lazy import avoids a core/channel_sc
    import cycle.
    """
    from .core import encode as _enc
    return _enc(u0.copy())


def _decode_recursive(lc: np.ndarray, u_hat: np.ndarray, frozen: np.ndarray, pos: int):
    """One SC recursion step over segment lc (Tal-Vardy Alg 2/5 core).

    Splits the segment into contiguous halves: the upper half decodes
    from f-combinations, then its partial sums condition the g-combinations
    for the lower half. Writes decisions into u_hat[pos:pos+m] and returns
    them. Returning decisions (not the partial codeword) is load-bearing:
    the parent's g-step re-encodes them itself via _partial_sums.
    """
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
    """Successive-cancellation decode: decide u_0..u_{N-1} in order.

    Each info bit is decided by the sign of its LLR given all previous
    decisions; frozen bits are forced to 0 without looking. Optimal among
    sequential decoders but fragile: one wrong early decision corrupts
    everything after it, which is exactly what list decoding fixes.
    """
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.uint8)
    _decode_recursive(llr.astype(float), u_hat, frozen, 0)
    return u_hat


def sc_llr(llr: np.ndarray, u_dec: np.ndarray, i: int) -> float:
    """LLR of bit i given past decisions u_dec[:i], iterative form.

    Same computation as _decode_recursive, but walks down only the single
    root-to-leaf path for bit i instead of decoding everything. SCL calls
    this once per path per bit, which is why this SCL costs O(L n^2):
    no LLR memoization across paths (the paper's Algs 6/14 add that).
    """
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
