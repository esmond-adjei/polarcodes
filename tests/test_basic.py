import numpy as np
from polar import encode, frozen_set_bec, sc_decode, awgn_llr


def test_encode_decode_noiseless():
    N, K = 16, 8
    frozen = frozen_set_bec(N, K)
    info = np.where(~frozen)[0]
    u = np.zeros(N, dtype=np.uint8)
    u[info] = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    x = encode(u)
    llr = np.where(x == 0, 10.0, -10.0)
    assert np.array_equal(sc_decode(llr, frozen), u)


def test_encode_involution():
    N = 8
    u = np.array([0, 0, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
    assert np.array_equal(encode(encode(u)), u)  # G_N^2 = I
