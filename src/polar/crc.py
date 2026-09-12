"""CRC helpers (CRC-16-CCITT) for CA-SCL."""
import numpy as np


def crc16_bits(msg: np.ndarray) -> np.ndarray:
    """CRC-16-CCITT (poly 0x1021, init 0xFFFF) of msg, MSB first.

    Bit-by-bit loop, fast enough: it runs once per block, not per
    decoded bit. Returns 16 bits to append after the message.
    """
    crc = 0xFFFF
    for b in msg:
        crc ^= int(b) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return np.array([(crc >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


def append_crc(msg: np.ndarray) -> np.ndarray:
    """Concatenate msg with its 16 CRC bits. The result is what occupies
    the K info positions, so a (K-16)-bit message becomes K bits on the
    channel. Callers must size messages as K - crc_len accordingly.
    """
    return np.concatenate([msg, crc16_bits(msg)])


def check_crc(msg_with_crc: np.ndarray) -> bool:
    """Recompute the CRC over all but the last 16 bits and compare.

    Returns False (not an exception) on mismatch, because a failed check
    is the normal "this path is wrong" signal in CA-SCL, not an error.
    """
    m, c = msg_with_crc[:-16], msg_with_crc[-16:]
    return bool(np.array_equal(crc16_bits(m), c))
