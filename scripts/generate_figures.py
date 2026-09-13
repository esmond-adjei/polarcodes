"""Generate paper-style figures: polarization, construction, and decoding.

Data shapes (principle-model-the-domain):
  Z[N]          : Bhattacharyya values in [0,1] for each synthetic channel
  m[N]          : GA mean LLRs, larger means more reliable
  frozen[N]     : bool mask, True=frozen
  curve         : list of (snr_db, ber, fer) per decoder

Run:  uv run python scripts/generate_figures.py --quick
      uv run python scripts/generate_figures.py --full   # larger N, more blocks
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Use non-interactive backend when running headless
plt.switch_backend("Agg")


def _bhattacharyya_vec(N: int, eps: float = 0.5) -> np.ndarray:
    n = int(np.log2(N))
    z = np.full(N, eps, dtype=float)
    step = 1
    for _ in range(n):
        nxt = np.empty(N)
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = z[i + j]
                nxt[i + j] = 2 * a - a * a
                nxt[i + j + step] = a * a
        z = nxt
        step *= 2
    return z


def _ga_means(N: int, K: int, design_snr_db: float) -> np.ndarray:
    from polar.core import awgn_construction  # noqa: used for sigma
    from polar.channel_sc import snr_to_sigma
    from polar.core import _phi, _phi_inv

    sigma = snr_to_sigma(design_snr_db, rate=K / N)
    m = np.full(N, 2.0 / sigma**2)
    step = 1
    n = int(np.log2(N))
    for _ in range(n):
        nxt = np.empty(N)
        for i in range(0, N, 2 * step):
            for j in range(step):
                a = m[i + j]
                nxt[i + j] = _phi_inv(1 - (1 - _phi(a)) ** 2)
                nxt[i + j + step] = 2 * a
        m = nxt
        step *= 2
    return m


# ------------------------------------------------------------------ fig 1
def fig_polarization_bec(out: str, N_list=(32, 128, 512, 1024)):
    """Arikan Fig. 4 style: sorted $Z(W_N^{(i)})$ as point cloud shows clustering to 0 or 1."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)
    axes = axes.ravel()
    for ax, N in zip(axes, N_list):
        z = np.sort(_bhattacharyya_vec(N, 0.5))
        ax.scatter(np.arange(N) / N, z, s=8, alpha=0.6, color="#4a7abc", edgecolors="none")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        near_good = np.mean(z < 0.01)
        near_bad = np.mean(z > 0.99)
        ax.set_title(f"$N={N}$, $\\epsilon=0.5$  ($Z<0.01$:{near_good:.0%}  $Z>0.99$:{near_bad:.0%})", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel("Sorted channel index $i/N$")
    for ax in axes[::2]:
        ax.set_ylabel("$Z(W_N^{(i)})$")
    fig.suptitle("Polarization of BEC: sorted Bhattacharyya parameters $Z$ (Arikan Fig. 4 style)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_polarization_histogram(out: str, N: int = 1024):
    """Histogram of $Z$ values at one block length, showing bimodal split."""
    z = _bhattacharyya_vec(N, 0.5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(z, bins=50, range=(0, 1), color="#4a7abc", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("$Z(W_N^{(i)})$")
    ax.set_ylabel("Count")
    ax.set_title(f"BEC polarization histogram $N={N}$, $\\epsilon=0.5$ (capacity $=0.5$)")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.text(0.02, ax.get_ylim()[1] * 0.9, f"Good ($Z<0.01$): {np.mean(z < 0.01):.1%}\nBad ($Z>0.99$): {np.mean(z > 0.99):.1%}",
            fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_frozen_pattern(out: str, N: int = 64, K: int = 32):
    """Visualize which positions are frozen (Arikan Fig. 9 style, small $N$)."""
    from polar.core import frozen_set_bec, awgn_construction
    z = _bhattacharyya_vec(N, 0.5)
    order_bec = np.argsort(z)
    frozen_bec = np.ones(N, dtype=bool)
    frozen_bec[order_bec[:K]] = False

    m = _ga_means(N, K, 1.0)
    order_ga = np.argsort(m)
    frozen_ga = np.ones(N, dtype=bool)
    frozen_ga[order_ga[-K:]] = False

    fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1]))
    for ax, frozen, label, vals, cmap in [
        (axes[0], frozen_bec, "BEC $\\epsilon=0.5$ frozen pattern", z, "Blues"),
        (axes[1], frozen_ga, "AWGN GA $E_b/N_0=1$ dB frozen pattern", m, "Oranges"),
    ]:
        colors = np.where(frozen, 0.15, 0.85)
        ax.bar(np.arange(N), np.ones(N), color=plt.get_cmap(cmap)(colors), edgecolor="white", linewidth=0.5)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=8)
        # annotate frozen vs info counts
        ax.text(0.01, 0.5, "frozen" if frozen[0] else "info", transform=ax.transAxes, fontsize=7, va="center", ha="left",
                color="white" if frozen[0] else "black", weight="bold")
    # bottom: sorted reliability curves with threshold (point clouds)
    axes[2].scatter(np.arange(N), np.sort(z), s=12, alpha=0.6, label="BEC $Z$ sorted (low=good)", color="#4a7abc")
    # normalize m for overlay
    m_sorted = np.sort(m)
    # scale m to [0,1] for visual comparison
    m_norm = (m_sorted - m_sorted.min()) / (m_sorted.max() - m_sorted.min() + 1e-12)
    axes[2].scatter(np.arange(N), m_norm, s=12, alpha=0.6, label="GA mean LLR sorted, normalized", color="#c97a2b")
    axes[2].axvline(N - K, color="red", linestyle="--", linewidth=1, label=f"Info threshold $K={K}$")
    axes[2].set_xlabel("Sorted synthetic channel index")
    axes[2].set_ylabel("Reliability (normalized)")
    axes[2].legend(fontsize=8, loc="center left")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Frozen-set construction $N={N}$, $K={K}$ ($R=1/2$)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_design_snr_effect(out: str, N: int = 256, K: int = 128):
    """How design $E_b/N_0$ shifts the frozen set (Tal-Vardy 2013 Fig. 2 intuition)."""
    snrs = [-1.0, 0.0, 1.0, 3.0]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    # top: sorted mean LLRs at each design SNR (point cloud per polarization request)
    for snr in snrs:
        m = _ga_means(N, K, snr)
        axes[0].scatter(np.arange(N), np.sort(m), s=10, alpha=0.55, label=f"design $E_b/N_0={snr:.0f}$ dB")
    axes[0].set_ylabel("Sorted mean LLR $m$")
    axes[0].set_title(f"GA construction vs design SNR $N={N}$, $R=1/2$")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")
    # bottom: Jaccard overlap of frozen sets vs 1 dB reference
    ref = set(np.argsort(_ga_means(N, K, 1.0))[-K:])
    overlaps = []
    for snr in snrs:
        cur = set(np.argsort(_ga_means(N, K, snr))[-K:])
        overlaps.append(len(ref & cur) / K)
    axes[1].bar([str(s) for s in snrs], overlaps, color="#6a9f6a", edgecolor="white")
    axes[1].set_ylabel("Overlap with 1 dB set")
    axes[1].set_xlabel("Design $E_b/N_0$ (dB)")
    axes[1].set_ylim(0, 1.02)
    for i, v in enumerate(overlaps):
        axes[1].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_butterfly(out: str, N: int = 8):
    """Arikan Fig. 5/9 style: butterfly stages for $N=8$, labeling $u \\to x$."""
    n = int(np.log2(N))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.invert_yaxis()
    # node positions: stage 0 = u, stage n = x
    for stage in range(n + 1):
        for i in range(N):
            ax.text(stage, i, f"${'u' if stage == 0 else 's' + str(stage) if stage < n else 'x'}_{i}$",
                    ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="circle,pad=0.25", facecolor="white", edgecolor="#4a7abc" if stage in (0, n) else "gray"))
    # draw butterfly edges for each stage
    for stage in range(n):
        step = 2 ** stage
        for i in range(0, N, 2 * step):
            for j in range(step):
                # upper: straight + xor diagonal
                x0, y0 = stage, i + j
                x1, y1 = stage + 1, i + j
                ax.plot([x0, x1], [y0, y1], color="#4a7abc", linewidth=1.2, alpha=0.7)
                # lower edge, with xor marker on upper destination
                ax.plot([x0, x1], [i + j + step, y1], color="#4a7abc", linewidth=0.8, alpha=0.4, linestyle="--")
                # xor dot at upper destination
                ax.plot(x1, y1, marker="o", color="#c97a2b", markersize=4)
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels([f"$u$" if i == 0 else f"$x$" if i == n else f"stage {i}" for i in range(n + 1)])
    ax.set_yticks([])
    ax.set_title(f"Polar butterfly $x = u F^{{\\otimes {n}}}$, $N={N}$ (Arikan Fig. 5/9 style)", fontsize=11)
    ax.text(0.5, -0.9, "Solid: pass-through   Dashed: XOR into upper   Orange dot: $\\oplus$", ha="center", fontsize=7, transform=ax.transData)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------- decoding
def _run_curves(N: int, K: int, snrs: list[float], decoders: list[tuple[str, int, int]], design_snr: float, blocks: int):
    from polar.core import awgn_construction
    from polar.sim import run_point
    frozen = awgn_construction(N, K, design_snr)
    results: dict[str, tuple[list[float], list[float]]] = {}
    for name, L, crc in decoders:
        bers, fers = [], []
        for s in snrs:
            b, f = run_point(N, K, frozen, s, decoder=name, L=L, n_blocks=blocks, crc_len=crc, seed=0)
            bers.append(max(b, 1e-6))
            fers.append(max(f, 1e-6))
            print(f"  {name} L={L} SNR {s}: BER {b:.2e} FER {f:.2e}")
        results[f"{name} L={L}" + ("+CRC" if crc else "")] = (bers, fers)
    return results, snrs


def fig_fer_vs_snr(out: str, quick: bool = True):
    """Tal-Vardy Fig. 1/5 style: FER vs $E_b/N_0$ for SC, SCL, CA-SCL."""
    if quick:
        N, K, snrs, blocks = 256, 128, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 80
        decoders = [("sc", 1, 0), ("scl", 4, 0), ("scl", 8, 0), ("ca-scl", 8, 16)]
    else:
        N, K, snrs, blocks = 1024, 512, [0.5, 1.0, 1.5, 2.0, 2.5], 200
        decoders = [("sc", 1, 0), ("scl", 2, 0), ("scl", 8, 0), ("scl", 32, 0), ("ca-scl", 32, 16)]
    results, snrs = _run_curves(N, K, snrs, decoders, design_snr=1.0, blocks=blocks)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for label, (bers, fers) in results.items():
        axes[0].semilogy(snrs, bers, "o-", label=label, linewidth=1.2, markersize=4)
        axes[1].semilogy(snrs, fers, "o-", label=label, linewidth=1.2, markersize=4)
    for ax, title in zip(axes, ["BER", "FER"]):
        ax.set_xlabel("$E_b/N_0$ (dB)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} $N={N}$, $K={K}$ ($R=1/2$)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("List decoding gains (Tal-Vardy Fig. 1/5 style)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_fer_vs_L(out: str, quick: bool = True):
    """FER vs list size $L$ at fixed $E_b/N_0$ (Tal-Vardy Fig. 1 insight, rotated)."""
    from polar.core import awgn_construction
    from polar.sim import run_point
    N, K = (256, 128) if quick else (1024, 512)
    snr = 2.0
    Ls = [1, 2, 4, 8, 16, 32] if not quick else [1, 2, 4, 8, 16]
    blocks = 80 if quick else 120
    frozen = awgn_construction(N, K, 1.0)
    fers_scl, fers_ca = [], []
    for L in Ls:
        _, f1 = run_point(N, K, frozen, snr, decoder="scl", L=L, n_blocks=blocks, seed=1)
        _, f2 = run_point(N, K, frozen, snr, decoder="ca-scl", L=L, n_blocks=blocks, crc_len=16, seed=1)
        fers_scl.append(max(f1, 5e-4))
        fers_ca.append(max(f2, 5e-4))
        print(f"  L={L}: SCL FER {f1:.3f}  CA-SCL FER {f2:.3f}")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(Ls, fers_scl, "o-", label="SCL", color="#4a7abc")
    ax.semilogy(Ls, fers_ca, "s-", label="CA-SCL (CRC-16)", color="#c97a2b")
    ax.set_xlabel("List size $L$")
    ax.set_ylabel("FER")
    ax.set_title(f"FER vs list size $N={N}$, $R=1/2$, $E_b/N_0={snr}$ dB ({blocks} blocks)")
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(x) for x in Ls])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description="Generate polar code figures")
    ap.add_argument("--quick", action="store_true", help="smaller N/blocks for fast run")
    ap.add_argument("--full", action="store_true", help="larger N/blocks (slower, closer to paper)")
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    quick = not args.full  # default quick unless --full
    if args.quick:
        quick = True
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # polarization + construction (fast, no Monte Carlo)
    fig_polarization_bec(os.path.join(outdir, "fig01_polarization_bec.png"))
    fig_polarization_histogram(os.path.join(outdir, "fig02_histogram.png"))
    fig_frozen_pattern(os.path.join(outdir, "fig03_frozen_pattern.png"))
    fig_design_snr_effect(os.path.join(outdir, "fig04_design_snr.png"))
    fig_butterfly(os.path.join(outdir, "fig05_butterfly.png"))

    # decoding curves (Monte Carlo, slower)
    fig_fer_vs_snr(os.path.join(outdir, "fig06_fer_vs_snr.png"), quick=quick)
    fig_fer_vs_L(os.path.join(outdir, "fig07_fer_vs_L.png"), quick=quick)

    print(f"All figures in {outdir}/")


if __name__ == "__main__":
    main()
