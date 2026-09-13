"""Monte-Carlo simulation harness for SC/SCL/CA-SCL."""
import numpy as np
from .core import encode
from .channel_sc import awgn_llr, sc_decode, sc_llr
from .scl import scl_decode, ca_scl_decode
from .crc import append_crc


def run_point(N, K, frozen, snr_db, decoder="sc", L=8, n_blocks=1000,
              payload_len=None, seed=0, crc_init=0xFFFF, progress=False):
    """Return (BER, FER, blocks) for one Eb/N0 point.

    ``K`` is the number of unfrozen positions. ``payload_len`` is the actual
    user-bit count. For ordinary SC/SCL it defaults to K. For Tal-Vardy's
    CRC-16 experiment use K=1040 and payload_len=1024, giving effective
    information rate 1024/N = 1/2 at N=2048.
    """
    rng = np.random.default_rng(seed)
    info_idx = np.where(~frozen)[0]
    if len(info_idx) != K:
        raise ValueError("frozen mask does not contain K information positions")
    if payload_len is None:
        payload_len = K
    if payload_len <= 0 or payload_len > K:
        raise ValueError("payload_len must be in [1, K]")

    ber_err = fer_err = 0
    iterator = range(n_blocks)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"{decoder}@{snr_db:g}dB", leave=False)

    # Eb/N0 is always defined against the user payload rate.
    rate = payload_len / N
    for _ in iterator:
        msg = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
        info = append_crc(msg, crc_init) if payload_len < K else msg
        if len(info) != K:
            raise ValueError("payload_len and K are inconsistent with CRC length")
        u = np.zeros(N, dtype=np.uint8)
        u[info_idx] = info
        llr = awgn_llr(encode(u), snr_db, rng, rate=rate)
        if decoder == "sc":
            uhat = sc_decode(llr, frozen)
        elif decoder == "scl":
            uhat = scl_decode(llr, frozen, L)[0]
        elif decoder == "ca-scl":
            uhat = ca_scl_decode(llr, frozen, info_idx, L, crc_init)
        else:
            raise ValueError(decoder)
        decoded = uhat[info_idx][:payload_len]
        err = decoded != msg
        ber_err += int(np.sum(err))
        fer_err += int(np.any(err))
    return ber_err / (n_blocks * payload_len), fer_err / n_blocks, n_blocks


def run_ml_bound_point(N, K, frozen, snr_db, n_blocks=10000, L=32,
                       payload_len=None, seed=0, crc_init=0xFFFF,
                       progress=False):
    """Estimate Tal--Vardy's empirical ML lower bound.

    For each SCL(L=32) frame error, compare the likelihood of the decoded
    codeword with the likelihood of the transmitted codeword. Count the event
    decoded likelihood > transmitted likelihood. This is the lower-bound
    construction described below Fig. 1 of the paper.
    """
    from .scl import scl_decode, _update_pm
    rng = np.random.default_rng(seed)
    info_idx = np.where(~frozen)[0]
    if payload_len is None:
        payload_len = K
    rate = payload_len / N
    events = 0
    failures = 0
    iterator = range(n_blocks)
    if progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"ML-bound@{snr_db:g}dB", leave=False)
    for _ in iterator:
        msg = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
        info = append_crc(msg, crc_init) if payload_len < K else msg
        u = np.zeros(N, dtype=np.uint8)
        u[info_idx] = info
        llr = awgn_llr(encode(u), snr_db, rng, rate=rate)
        cand = scl_decode(llr, frozen, L)[0]
        if not np.array_equal(cand[info_idx][:payload_len], msg):
            failures += 1
            # Reconstruct the two accumulated negative log-likelihoods.
            pm_c = pm_t = 0.0
            uc = np.zeros(N, dtype=np.uint8)
            for i in range(N):
                Lc = sc_llr(llr, uc, i)
                pm_c = _update_pm(pm_c, float(Lc), int(cand[i]))
                uc[i] = cand[i]
            ut = np.zeros(N, dtype=np.uint8)
            for i in range(N):
                Lt = sc_llr(llr, ut, i)
                pm_t = _update_pm(pm_t, float(Lt), int(u[i]))
                ut[i] = u[i]
            if pm_c < pm_t:
                events += 1
    return events / n_blocks, failures / n_blocks, n_blocks
