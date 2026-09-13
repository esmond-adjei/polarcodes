"""Successive-cancellation list decoding (Tal & Vardy 2012).

Vectorized implementation with O(L·N·log N) LLR work in NumPy.

The previous reference version called ``sc_llr`` per path per bit, which
re-derives the whole decoding tree from scratch: O(L·N²) Python-level work
(~190 s/block at N=2048, L=32). This version instead recurses over
contiguous segments exactly like SC decoding, computing each segment's
f/g LLRs once per surviving path set with NumPy vectorization:

* f (check node, exact box-plus): logaddexp(0,a+b) - logaddexp(a,b)
* g (variable node): b + (1-2·u)·a, with u the partial sums
* path metric: PM += log(1 + exp(-(1-2·u)·L)) via logaddexp

Fork/prune semantics are identical to Tal-Vardy Algs 16-18: each info bit
forks survivors in two, frozen bits extend in one, keep the L best by PM.
Candidate ordering matches the old scalar loop (interleaved per-parent,
stable sort) so results are bit-identical up to floating-point ufunc
ordering (same ufunc, same order) apart from vanishingly rare exact ties.
"""
import numpy as np
from .crc import check_crc


def _update_pm(pm: float, L: float, u: int) -> float:
    """Stable LLR-domain negative log-likelihood increment."""
    return pm + float(np.logaddexp(0.0, -(1 - 2 * int(u)) * L))


def _boxplus_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def _encode_batch_inplace(U: np.ndarray) -> np.ndarray:
    """In-place polar butterfly over axis 1 for a batch of rows.

    U shape (P, H) with H a power of two, dtype uint8. Applies the same
    stage order as core.encode so that per-row results equal encode(row).
    """
    P, H = U.shape
    if H <= 1:
        return U
    step = 1
    while step < H:
        # U is C-contiguous here (we always pass a .copy()), so this
        # reshape is a view and the XOR writes back into U.
        Vr = U.reshape(P, -1, 2 * step)
        # LHS and RHS slices are disjoint, so in-place is safe.
        Vr[:, :, :step] ^= Vr[:, :, step:]
        step <<= 1
    return U


def _scl_decode_with_pm(llr: np.ndarray, frozen: np.ndarray, L: int):
    """Inner worker returning (paths_list, pm_array) best-first."""
    N = int(len(llr))
    if N & (N - 1):
        raise ValueError("N must be a power of two")
    frozen = np.asarray(frozen, dtype=bool)
    llr0 = np.asarray(llr, dtype=float)

    U_init = np.zeros((1, N), dtype=np.uint8)
    PM_init = np.zeros(1, dtype=float)
    llr_seg0 = np.tile(llr0, (1, 1))

    def recurse(llr_seg: np.ndarray, pos: int, U: np.ndarray,
                PM: np.ndarray):
        P, seg = llr_seg.shape
        if seg == 1:
            li = llr_seg[:, 0]
            if frozen[pos]:
                PM = PM + np.logaddexp(0.0, -li)
                U[:, pos] = 0
                return U, PM, np.arange(P, dtype=int)
            # info bit: fork in two, keep L best.
            # Interleaved candidate order [p0-0, p0-1, p1-0, p1-1, ...]
            # matches the old scalar loop so stable-sort ties agree.
            pen0 = np.logaddexp(0.0, -li)
            pen1 = np.logaddexp(0.0, li)
            PM0 = PM + pen0
            PM1 = PM + pen1
            n_cand = 2 * P
            cand_PM = np.empty(n_cand, dtype=float)
            cand_PM[0::2] = PM0
            cand_PM[1::2] = PM1
            cand_parent = np.repeat(np.arange(P, dtype=int), 2)
            # stable sort keeps old tie-breaking (u=0 before u=1 per parent)
            order = np.argsort(cand_PM, kind="stable")[:L]
            sel_parent = cand_parent[order]
            U_new = U[sel_parent].copy()
            # bit = order parity: even index -> u=0, odd -> u=1
            U_new[:, pos] = (order & 1).astype(np.uint8)
            PM_new = cand_PM[order]
            return U_new, PM_new, sel_parent
        half = seg // 2
        f_out = _boxplus_vec(llr_seg[:, :half], llr_seg[:, half:])
        U1, PM1, map_left = recurse(f_out, pos, U, PM)
        # NOTE: must always gather through map_left, even when the path
        # count is unchanged. Pruning can keep P rows while replacing one
        # parent with a duplicate of another (e.g. 2 -> 2 with map [0,0]),
        # so reusing llr_seg unchanged would mix LLRs from killed paths.
        llr_expanded = llr_seg[map_left]
        # Partial sums of the decoded left half, batched in NumPy.
        left = U1[:, pos:pos + half].copy()
        _encode_batch_inplace(left)
        a2 = llr_expanded[:, :half]
        b2 = llr_expanded[:, half:]
        g_out = b2 + (1.0 - 2.0 * left.astype(float)) * a2
        U2, PM2, map_right = recurse(g_out, pos + half, U1, PM1)
        final_map = map_left[map_right]
        return U2, PM2, final_map

    U_final, PM_final, _ = recurse(llr_seg0, 0, U_init, PM_init)
    order = np.argsort(PM_final, kind="stable")
    U_sorted = U_final[order]
    PM_sorted = PM_final[order]
    return [r.copy() for r in U_sorted], PM_sorted


def scl_decode(llr: np.ndarray, frozen: np.ndarray, L: int) -> list[np.ndarray]:
    """Return up to L candidate u-vectors, ordered by path likelihood."""
    if L < 1:
        raise ValueError("L must be >= 1")
    paths, _ = _scl_decode_with_pm(np.asarray(llr), np.asarray(frozen), int(L))
    return paths


def ca_scl_decode(llr: np.ndarray, frozen: np.ndarray, info_idx: np.ndarray,
                  L: int, crc_init: int = 0xFFFF) -> np.ndarray:
    """CRC-aided final selection: filter valid paths, then choose best PM."""
    paths = scl_decode(llr, frozen, L)
    info_idx = np.asarray(info_idx)
    valid = [p for p in paths if check_crc(p[info_idx], crc_init)]
    return valid[0] if valid else paths[0]


def path_metric(llr: np.ndarray, u: np.ndarray) -> float:
    """Negative log-likelihood of a full u-vector in one SC pass.

    O(N·log N) with NumPy vectorization. Replaces the old O(N²) loop of
    N independent sc_llr calls in the ML-bound estimator. Uses the same
    exact f/g and logaddexp PM update as the list decoder, so
    ``path_metric(llr, u)`` equals the PM the list decoder would assign
    to that path.
    """
    llr_seg = np.asarray(llr, dtype=float)
    u = np.asarray(u, dtype=np.uint8)
    N = len(llr_seg)
    if N & (N - 1):
        raise ValueError("N must be a power of two")

    def rec(lc: np.ndarray, pos: int) -> float:
        m = len(lc)
        if m == 1:
            return float(np.logaddexp(0.0, -(1 - 2 * int(u[pos])) * float(lc[0])))
        half = m // 2
        f_part = _boxplus_vec(lc[:half], lc[half:])
        pm_left = rec(f_part, pos)
        s0 = left_partial_sums(u[pos:pos + half].copy())
        g_part = lc[half:] + (1.0 - 2.0 * s0.astype(float)) * lc[:half]
        pm_right = rec(g_part, pos + half)
        return pm_left + pm_right

    return rec(llr_seg, 0)


def left_partial_sums(u0: np.ndarray) -> np.ndarray:
    """Polar transform of one half-segment (partial sums for g-nodes)."""
    from .core import encode as _enc
    return _enc(np.asarray(u0, dtype=np.uint8).copy())
