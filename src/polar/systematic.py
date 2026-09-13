"""Systematic polar encoding."""
from functools import lru_cache
import numpy as np
from .core import encode


@lru_cache(maxsize=8)
def _systematic_inv_matrix(N: int, info_key: tuple) -> np.ndarray:
    """Inverse of the information submatrix over GF(2), cached.

    The elimination depends only on (N, info_idx), not on the message, so
    cache it: repeated encodes with the same frozen set pay O(K^2) matvec
    instead of O(K^3) Gauss-Jordan each call.
    """
    from .core import kron_power
    info_idx = np.asarray(info_key, dtype=int)
    K = len(info_idx)
    n = int(np.log2(N))
    G = kron_power(n)
    A = G[np.ix_(info_idx, info_idx)].T.copy().astype(np.uint8)
    aug = np.concatenate(
        [A, np.eye(K, dtype=np.uint8)], axis=1).astype(np.uint8)
    row = 0
    for col in range(K):
        piv = next((r for r in range(row, K) if aug[r, col]), None)
        if piv is None:
            raise ValueError("information submatrix is singular")
        if piv != row:
            aug[[row, piv]] = aug[[piv, row]]
        for r in range(K):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    if not np.array_equal(aug[:, :K],
                           np.eye(K, dtype=np.uint8)):
        raise ValueError("information submatrix is singular")
    return aug[:, K:].copy()


def systematic_encode(info: np.ndarray, info_idx: np.ndarray, N: int) -> np.ndarray:
    """Encode so the unfrozen codeword coordinates equal ``info``.

    For the conventional Arikan transform used in this repository, the
    systematic vector is obtained by solving the corresponding submatrix of
    the generator transform over GF(2), then applying the ordinary encoder.
    The GF(2) inverse depends only on ``(N, info_idx)`` and is cached, so
    repeated calls with the same frozen set cost one O(K^2) matvec each.
    """
    info = np.asarray(info, dtype=np.uint8)
    info_idx = np.asarray(info_idx, dtype=int)
    if len(info) != len(info_idx):
        raise ValueError("info and info_idx must have the same length")
    if N & (N - 1):
        raise ValueError("N must be a power of two")
    inv = _systematic_inv_matrix(N, tuple(int(i) for i in info_idx))
    u_info = (inv.astype(np.int32) @ info.astype(np.int32)) & 1
    u = np.zeros(N, dtype=np.uint8)
    u[info_idx] = u_info.astype(np.uint8)
    return encode(u)
