"""Systematic polar encoding."""
import numpy as np
from .core import encode


def systematic_encode(info: np.ndarray, info_idx: np.ndarray, N: int) -> np.ndarray:
    """Encode so the unfrozen codeword coordinates equal ``info``.

    For the conventional Arikan transform used in this repository, the
    systematic vector is obtained by solving the corresponding submatrix of
    the generator transform over GF(2), then applying the ordinary encoder.
    This implementation is deliberately straightforward and intended for
    reproduction/reference experiments.
    """
    info = np.asarray(info, dtype=np.uint8)
    info_idx = np.asarray(info_idx, dtype=int)
    if len(info) != len(info_idx):
        raise ValueError("info and info_idx must have the same length")
    if N & (N - 1):
        raise ValueError("N must be a power of two")
    n = int(np.log2(N))
    from .core import kron_power
    G = kron_power(n)
    A = G[np.ix_(info_idx, info_idx)].copy()
    rhs = info.copy()
    # Gauss-Jordan over GF(2): A^T * u_info = rhs under row-vector encoding.
    A = A.T.copy()
    aug = np.concatenate([A, rhs[:, None]], axis=1).astype(np.uint8)
    row = 0
    for col in range(len(info_idx)):
        piv = next((r for r in range(row, len(info_idx)) if aug[r, col]), None)
        if piv is None:
            raise ValueError("information submatrix is singular")
        if piv != row:
            aug[[row, piv]] = aug[[piv, row]]
        for r in range(len(info_idx)):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        row += 1
    u = np.zeros(N, dtype=np.uint8)
    u[info_idx] = aug[:, -1]
    return encode(u)
