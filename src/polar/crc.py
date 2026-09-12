"""CRC helpers (CRC-16-CCITT) for CA-SCL."""
import numpy as np


def crc16_bits(msg: np.ndarray) -> np.ndarray:
    crc = 0xFFFF
    for b in msg:
        crc ^= int(b) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return np.array([(crc >> (15 - i)) & 1 for i in range(16)], dtype=np.uint8)


def append_crc(msg: np.ndarray) -> np.ndarray:
    return np.concatenate([msg, crc16_bits(msg)])


def check_crc(msg_with_crc: np.ndarray) -> bool:
    m, c = msg_with_crc[:-16], msg_with_crc[-16:]
    return bool(np.array_equal(crc16_bits(m), c))
