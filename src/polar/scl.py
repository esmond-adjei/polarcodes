"""Stage 3/4 — SCL and CRC-aided SCL (Tal & Vardy 2012).

Path metric (LLR domain): PM += log(1 + exp(-(1-2u)·L)).
"""
import numpy as np
from .channel_sc import sc_llr
from .crc import check_crc


def _update_pm(pm: float, L: float, u: int) -> float:
    """Add bit u's penalty to path metric pm given its LLR L.

    Penalty = log(1 + exp(-(1-2u)L)): ~0 when u agrees with L's sign,
    ~|L| when it contradicts it. So pm is the total log-likelihood cost
    of the path's decisions, and ranking by pm ranks by likelihood.
    Frozen bits still pay the u=0 penalty, which keeps paths comparable.
    """
    return pm + float(np.log1p(np.exp(-(1 - 2 * int(u)) * L)))


def scl_decode(llr: np.ndarray, frozen: np.ndarray, L: int) -> list[np.ndarray]:
    """List-decode to the L most likely u vectors, best first.

    Follows Tal-Vardy Algs 16-18: each info bit forks every survivor in
    two (frozen bits extend without forking), all forks are ranked by
    path metric, and only the L best continue. L=1 reduces to SC. The
    full-vector copy per fork is the O(L n^2) bottleneck the paper's
    lazy-copy structure (Algs 8-13) removes.
    """
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
    """Return the best list path whose info bits pass the CRC.

    Why: the ML path is usually somewhere in the list but not ranked
    first, and the CRC identifies it. Falls back to the best path when
    nothing passes (all-wrong list), so behavior degrades to plain SCL
    instead of crashing. info_idx must match the layout sim.py used when
    appending the CRC, or every check fails and you silently get SCL.
    """
    for c in scl_decode(llr, frozen, L):
        if check_crc(c[info_idx]):
            return c
    return scl_decode(llr, frozen, L)[0]
