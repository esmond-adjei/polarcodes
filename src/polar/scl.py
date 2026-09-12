"""Stage 3/4 — SCL and CRC-aided SCL (Tal & Vardy 2012).

Path metric (LLR domain): PM += log(1 + exp(-(1-2u)·L)).
"""
import numpy as np
from .channel_sc import sc_llr
from .crc import check_crc


def _update_pm(pm: float, L: float, u: int) -> float:
    return pm + float(np.log1p(np.exp(-(1 - 2 * int(u)) * L)))


def scl_decode(llr: np.ndarray, frozen: np.ndarray, L: int) -> list[np.ndarray]:
    """Return up to L candidate u_hat vectors (best first)."""
    paths = [{"u": np.zeros(len(llr), dtype=np.uint8), "pm": 0.0}]
    for i in range(len(llr)):
        cands = []
        for p in paths:
            Li = sc_llr(llr, p["u"], i)
            if frozen[i]:
                p["u"][i] = 0
                p["pm"] = _update_pm(p["pm"], Li, 0)
                cands.append(p)
            else:
                for b in (0, 1):
                    q = {"u": p["u"].copy(), "pm": _update_pm(p["pm"], Li, b)}
                    q["u"][i] = b
                    cands.append(q)
        cands.sort(key=lambda d: d["pm"])
        paths = cands[:L]
    paths.sort(key=lambda d: d["pm"])
    return [p["u"] for p in paths]


def ca_scl_decode(llr: np.ndarray, frozen: np.ndarray, info_idx: np.ndarray,
                  L: int) -> np.ndarray:
    """Pick best CRC-passing path; fall back to best path."""
    for c in scl_decode(llr, frozen, L):
        if check_crc(c[info_idx]):
            return c
    return scl_decode(llr, frozen, L)[0]
