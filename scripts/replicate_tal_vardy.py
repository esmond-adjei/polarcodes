"""Reproduce the main Tal--Vardy 2012 AWGN experiment.

Paper target: N=2048, effective rate 1/2, design Eb/N0=2 dB,
L in {1,2,4,8,16,32}. The paper's construction is described in [4],
Tal & Vardy (2013). ``--construction tv-mc`` is the default auditable
numerical density-evolution approximation; ``ga`` is a faster alternative.
"""
import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt
from polar import awgn_construction
from polar.construction import tv_mc_construction
from polar.sim import run_point, run_ml_bound_point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--K", type=int, default=1024,
                    help="unfrozen bits for ordinary SCL")
    ap.add_argument("--design-snr", type=float, default=2.0)
    ap.add_argument("--snrs", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--blocks", type=int, default=10000)
    ap.add_argument("--L", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--construction", choices=["tv-mc", "ga"], default="tv-mc")
    ap.add_argument("--construction-samples", type=int, default=4096)
    ap.add_argument("--construction-seed", type=int, default=12345)
    ap.add_argument("--ml-bound", action="store_true",
                    help="estimate the paper's empirical ML lower bound using L=32 failures")
    ap.add_argument("--crc", action="store_true",
                    help="also run the paper's CRC-16 experiment: K=1040, payload=1024")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--out", default="results/tal_vardy_fig1.png")
    ap.add_argument("--csv", default="results/tal_vardy_results.csv")
    a = ap.parse_args()

    if a.construction == "tv-mc":
        frozen = tv_mc_construction(a.N, a.K, a.design_snr,
                                     a.construction_samples, a.construction_seed)
    else:
        frozen = awgn_construction(a.N, a.K, a.design_snr)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    rows = []
    for L in a.L:
        dec = "sc" if L == 1 else "scl"
        for snr in a.snrs:
            ber, fer, blocks = run_point(a.N, a.K, frozen, snr, decoder=dec,
                                         L=L, n_blocks=a.blocks,
                                         payload_len=a.K, seed=a.seed + int(1000 * snr) + L,
                                         progress=True)
            rows.append((dec, L, snr, ber, fer, blocks))
            print(f"{dec:5s} L={L:2d} Eb/N0={snr:.2f} dB BER={ber:.3e} FER={fer:.3e}")

    if a.crc:
        crc_K = a.K + 16
        if crc_K > a.N:
            raise ValueError("K+16 must be <= N for the CRC experiment")
        if a.construction == "tv-mc":
            frozen_crc = tv_mc_construction(a.N, crc_K, a.design_snr,
                                            a.construction_samples, a.construction_seed)
        else:
            frozen_crc = awgn_construction(a.N, crc_K, a.design_snr)
        for snr in a.snrs:
            ber, fer, blocks = run_point(a.N, crc_K, frozen_crc, snr,
                                         decoder="ca-scl", L=max(a.L), n_blocks=a.blocks,
                                         payload_len=a.K, seed=a.seed + int(1000 * snr) + 999,
                                         progress=True)
            rows.append(("ca-scl", max(a.L), snr, ber, fer, blocks))
            print(f"ca-scl L={max(a.L):2d} Eb/N0={snr:.2f} dB BER={ber:.3e} FER={fer:.3e}")

    if a.ml_bound:
        for snr in a.snrs:
            lb, scl_fer, blocks = run_ml_bound_point(
                a.N, a.K, frozen, snr, n_blocks=a.blocks, L=32,
                payload_len=a.K, seed=a.seed + int(1000 * snr) + 4242, progress=True)
            rows.append(("ml-bound", 32, snr, np.nan, lb, blocks))
            print(f"ml-bound L=32 Eb/N0={snr:.2f} dB lower-bound FER={lb:.3e} (SCL FER={scl_fer:.3e})")

    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["decoder", "L", "EbN0_dB", "BER", "FER", "blocks"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    for dec in sorted(set(r[0] for r in rows)):
        for L in sorted(set(r[1] for r in rows if r[0] == dec)):
            rr = [r for r in rows if r[0] == dec and r[1] == L]
            ax.semilogy([r[2] for r in rr], [r[4] for r in rr], "o-",
                        label=f"{dec}, L={L}")
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel("Word / frame error rate")
    ax.set_title(f"Polar N={a.N}, effective rate 1/2, design={a.design_snr:g} dB")
    ax.grid(True, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=180)
    print("saved", a.out)
    print("saved", a.csv)


if __name__ == "__main__":
    main()
