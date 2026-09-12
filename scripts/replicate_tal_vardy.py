"""Replicate Tal & Vardy (2012) style curves: N=1024/2048, R=1/2, SC vs SCL vs CA-SCL."""
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from polar import awgn_construction
from polar.sim import run_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--blocks", type=int, default=200)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--design-snr", type=float, default=1.0)
    ap.add_argument("--out", default="results/tal_vardy.png")
    a = ap.parse_args()

    frozen = awgn_construction(a.N, a.K, a.design_snr)
    cfgs = [("sc", 1, 0), ("scl", a.L, 0), ("ca-scl", a.L, 16)]
    res = {}
    for dec, L, crc in cfgs:
        bers, fers = [], []
        for s in a.snrs:
            b, f = run_point(a.N, a.K, frozen, s, decoder=dec, L=L,
                             n_blocks=a.blocks, crc_len=crc)
            bers.append(b); fers.append(f)
            print(f"{dec} L={L} SNR={s}: BER={b:.2e} FER={f:.2e}")
        res[dec] = (bers, fers)
    import os
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for dec, (bers, fers) in res.items():
        ax[0].semilogy(a.snrs, bers, "o-", label=dec)
        ax[1].semilogy(a.snrs, fers, "o-", label=dec)
    ax[0].set(xlabel="Eb/N0 (dB)", ylabel="BER", title=f"Polar N={a.N} K={a.K}")
    ax[1].set(xlabel="Eb/N0 (dB)", ylabel="FER", title="Tal-Vardy style replication")
    for x in ax: x.legend(); x.grid(True, which="both")
    fig.tight_layout(); fig.savefig(a.out, dpi=150)
    print("saved", a.out)


if __name__ == "__main__":
    main()
