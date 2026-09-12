"""Simulation harness: BER/FER over AWGN for SC / SCL / CA-SCL."""
import numpy as np
from tqdm import tqdm
from .core import encode
from .channel_sc import awgn_llr
from .channel_sc import sc_decode
from .scl import scl_decode, ca_scl_decode


def run_point(N, K, frozen, snr_db, decoder="sc", L=8, n_blocks=200,
              crc_len=0, seed=0):
    rng = np.random.default_rng(seed)
    info_idx = np.where(~frozen)[0]
    assert len(info_idx) == K
    ber_err = ber_tot = fer_err = 0
    for _ in tqdm(range(n_blocks), leave=False, desc=f"{decoder}@{snr_db}dB"):
        msg = rng.integers(0, 2, size=K - crc_len).astype(np.uint8) if crc_len else \
            rng.integers(0, 2, size=K).astype(np.uint8)
        if crc_len:
            from .crc import append_crc
            msg = append_crc(msg)
        u = np.zeros(N, dtype=np.uint8)
        u[info_idx] = msg
        llr = awgn_llr(encode(u), snr_db, rng, rate=K / N)
        if decoder == "sc":
            uhat = sc_decode(llr, frozen)
        elif decoder == "scl":
            uhat = scl_decode(llr, frozen, L)[0]
        elif decoder == "ca-scl":
            uhat = ca_scl_decode(llr, frozen, info_idx, L)
        else:
            raise ValueError(decoder)
        mhat = uhat[info_idx]
        ber_err += np.sum(mhat[:len(msg)] != msg)
        ber_tot += len(msg)
        fer_err += int(np.any(mhat[:len(msg)] != msg))
    return ber_err / ber_tot, fer_err / n_blocks
