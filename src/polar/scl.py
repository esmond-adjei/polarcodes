"""Successive-cancellation list decoding (Tal & Vardy 2012)."""
import numpy as np
from .channel_sc import sc_llr
from .crc import check_crc


def _update_pm(pm: float, L: float, u: int) -> float:
    """Stable LLR-domain negative log-likelihood increment."""
    return pm + float(np.logaddexp(0.0, -(1 - 2 * int(u)) * L))


def scl_decode(llr: np.ndarray, frozen: np.ndarray, L: int) -> list[np.ndarray]:
    """Return up to L candidate u-vectors, ordered by path likelihood."""
    if L < 1:
        raise ValueError("L must be >= 1")
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
                  L: int, crc_init: int = 0xFFFF) -> np.ndarray:
    """CRC-aided final selection: filter valid paths, then choose best PM."""
    paths = scl_decode(llr, frozen, L)
    valid = [p for p in paths if check_crc(p[info_idx], crc_init)]
    return valid[0] if valid else paths[0]
