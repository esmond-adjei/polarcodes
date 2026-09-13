"""CRC-16-CCITT helpers used by the Tal-Vardy CRC-aided construction."""
import numpy as np

POLY = 0x1021
WIDTH = 16
MASK = (1 << WIDTH) - 1


def crc16_bits(msg: np.ndarray, init: int = 0xFFFF) -> np.ndarray:
    """CRC-16/CCITT bitstream, MSB first.

    The Tal-Vardy paper specifies the CRC polynomial through Peterson & Weldon
    rather than fixing an initialization convention. Therefore the init value
    is explicit and configurable; 0xFFFF is the common CRC-CCITT convention.
    """
    crc = int(init) & MASK
    for b in np.asarray(msg, dtype=np.uint8):
        bit = int(b)
        top = ((crc >> 15) & 1) ^ bit
        crc = (crc << 1) & MASK
        if top:
            crc ^= POLY
    return np.array([(crc >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


def append_crc(msg: np.ndarray, init: int = 0xFFFF) -> np.ndarray:
    return np.concatenate([np.asarray(msg, dtype=np.uint8), crc16_bits(msg, init)])


def check_crc(msg_with_crc: np.ndarray, init: int = 0xFFFF) -> bool:
    msg_with_crc = np.asarray(msg_with_crc, dtype=np.uint8)
    if len(msg_with_crc) < 16:
        return False
    return bool(np.array_equal(crc16_bits(msg_with_crc[:-16], init), msg_with_crc[-16:]))
