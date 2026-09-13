"""Reproduce the main Tal--Vardy 2012 AWGN experiment.

Paper target: N=2048, effective rate 1/2, design Eb/N0=2 dB,
L in {1,2,4,8,16,32}. The paper's construction is described in [4],
Tal & Vardy (2013).

--construction tv-mc is the default auditable numerical density-evolution
approximation; --construction ga is a faster alternative.

Use --workers to run independent simulation points in parallel:

    python reproduce_tal_vardy.py --workers 8

Each (decoder, L, SNR) point is an independent process, so this can
substantially reduce wall-clock time on multi-core machines.
"""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np

from polar import awgn_construction
from polar.construction import tv_mc_construction
from polar.sim import run_point, run_ml_bound_point


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_WORKER_FROZEN = None


def _init_worker(frozen):
    """Initialize immutable construction data once per worker process."""
    global _WORKER_FROZEN
    _WORKER_FROZEN = frozen


# ---------------------------------------------------------------------------
# Worker jobs
# ---------------------------------------------------------------------------

def _run_simulation_job(job):
    """Run one ordinary SC/SCL simulation point."""
    (
        decoder,
        L,
        snr,
        N,
        K,
        n_blocks,
        payload_len,
        seed,
    ) = job

    ber, fer, blocks = run_point(
        N,
        K,
        _WORKER_FROZEN,
        snr,
        decoder=decoder,
        L=L,
        n_blocks=n_blocks,
        payload_len=payload_len,
        seed=seed,
        progress=False,
    )

    return {
        "decoder": decoder,
        "L": L,
        "snr": snr,
        "ber": ber,
        "fer": fer,
        "blocks": blocks,
    }


def _run_crc_job(job):
    """Run one CRC-aided SCL simulation point."""
    (
        L,
        snr,
        N,
        crc_K,
        payload_len,
        n_blocks,
        seed,
    ) = job

    ber, fer, blocks = run_point(
        N,
        crc_K,
        _WORKER_FROZEN,
        snr,
        decoder="ca-scl",
        L=L,
        n_blocks=n_blocks,
        payload_len=payload_len,
        seed=seed,
        progress=False,
    )

    return {
        "decoder": "ca-scl",
        "L": L,
        "snr": snr,
        "ber": ber,
        "fer": fer,
        "blocks": blocks,
    }


def _run_ml_bound_job(job):
    """Run one empirical ML lower-bound point."""
    (
        snr,
        N,
        K,
        n_blocks,
        L,
        payload_len,
        seed,
    ) = job

    lb, scl_fer, blocks = run_ml_bound_point(
        N,
        K,
        _WORKER_FROZEN,
        snr,
        n_blocks=n_blocks,
        L=L,
        payload_len=payload_len,
        seed=seed,
        progress=False,
    )

    return {
        "decoder": "ml-bound",
        "L": L,
        "snr": snr,
        "ber": np.nan,
        "fer": lb,
        "blocks": blocks,
        "scl_fer": scl_fer,
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_construction(args, K):
    if args.construction == "tv-mc":
        return tv_mc_construction(
            args.N,
            K,
            args.design_snr,
            args.construction_samples,
            args.construction_seed,
        )

    return awgn_construction(
        args.N,
        K,
        args.design_snr,
    )


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------

def run_jobs(jobs, frozen, workers):
    """Execute jobs serially or across worker processes."""
    if not jobs:
        return []

    # Serial mode preserves the old behavior as closely as possible.
    if workers == 1:
        _init_worker(frozen)

        results = []

        for job in jobs:
            kind = job[0]

            if kind == "sim":
                result = _run_simulation_job(job[1:])
            elif kind == "crc":
                result = _run_crc_job(job[1:])
            elif kind == "ml-bound":
                result = _run_ml_bound_job(job[1:])
            else:
                raise ValueError(f"Unknown job type: {kind}")

            results.append(result)

            _print_result(result)

        return results

    # Parallel mode.
    #
    # The construction is transferred to each worker once through the
    # initializer rather than being serialized for every individual job.
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(frozen,),
    ) as executor:

        futures = {}

        for job in jobs:
            kind = job[0]

            if kind == "sim":
                future = executor.submit(
                    _run_simulation_job,
                    job[1:],
                )
            elif kind == "crc":
                future = executor.submit(
                    _run_crc_job,
                    job[1:],
                )
            elif kind == "ml-bound":
                future = executor.submit(
                    _run_ml_bound_job,
                    job[1:],
                )
            else:
                raise ValueError(f"Unknown job type: {kind}")

            futures[future] = job

        results = []

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _print_result(result)

        return results


def _print_result(result):
    decoder = result["decoder"]
    L = result["L"]
    snr = result["snr"]
    ber = result["ber"]
    fer = result["fer"]

    if decoder == "ml-bound":
        print(
            f"ml-bound L={L:2d} "
            f"Eb/N0={snr:.2f} dB "
            f"lower-bound FER={fer:.3e} "
            f"(SCL FER={result['scl_fer']:.3e})"
        )
        return

    print(
        f"{decoder:7s} "
        f"L={L:2d} "
        f"Eb/N0={snr:.2f} dB "
        f"BER={ber:.3e} "
        f"FER={fer:.3e}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--N", type=int, default=2048)

    ap.add_argument(
        "--K",
        type=int,
        default=1024,
        help="unfrozen bits for ordinary SCL",
    )

    ap.add_argument(
        "--design-snr",
        type=float,
        default=2.0,
    )

    ap.add_argument(
        "--snrs",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0],
    )

    ap.add_argument(
        "--blocks",
        type=int,
        default=10000,
    )

    ap.add_argument(
        "--L",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "number of parallel simulation worker processes; "
            "1 disables multiprocessing"
        ),
    )

    ap.add_argument(
        "--construction",
        choices=["tv-mc", "ga"],
        default="tv-mc",
    )

    ap.add_argument(
        "--construction-samples",
        type=int,
        default=4096,
    )

    ap.add_argument(
        "--construction-seed",
        type=int,
        default=12345,
    )

    ap.add_argument(
        "--ml-bound",
        action="store_true",
        help=(
            "estimate the paper's empirical ML lower bound "
            "using L=32 failures"
        ),
    )

    ap.add_argument(
        "--crc",
        action="store_true",
        help=(
            "also run the paper's CRC-16 experiment: "
            "K=1040, payload=1024"
        ),
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260913,
    )

    ap.add_argument(
        "--out",
        default="results/tal_vardy_fig1.png",
    )

    ap.add_argument(
        "--csv",
        default="results/tal_vardy_results.csv",
    )

    a = ap.parse_args()

    if a.workers < 1:
        ap.error("--workers must be >= 1")

    if a.blocks < 1:
        ap.error("--blocks must be >= 1")

    if any(L < 1 for L in a.L):
        ap.error("all L values must be >= 1")

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Main construction
    # ------------------------------------------------------------------

    print(
        f"Building {a.construction} construction "
        f"(N={a.N}, K={a.K}, design={a.design_snr:g} dB)..."
    )

    frozen = build_construction(a, a.K)

    # ------------------------------------------------------------------
    # Build ordinary SC/SCL jobs
    # ------------------------------------------------------------------

    jobs = []

    for L in a.L:
        decoder = "sc" if L == 1 else "scl"

        for snr in a.snrs:
            seed = (
                a.seed
                + int(1000 * snr)
                + L
            )

            jobs.append(
                (
                    "sim",
                    decoder,
                    L,
                    snr,
                    a.N,
                    a.K,
                    a.blocks,
                    a.K,
                    seed,
                )
            )

    print(
        f"Running {len(jobs)} ordinary simulation points "
        f"with {a.workers} worker(s)..."
    )

    rows = run_jobs(
        jobs,
        frozen,
        a.workers,
    )

    # ------------------------------------------------------------------
    # CRC experiment
    # ------------------------------------------------------------------

    if a.crc:
        crc_K = a.K + 16

        if crc_K > a.N:
            raise ValueError(
                "K+16 must be <= N for the CRC experiment"
            )

        print(
            f"Building {a.construction} CRC construction "
            f"(N={a.N}, K={crc_K}, design={a.design_snr:g} dB)..."
        )

        frozen_crc = build_construction(a, crc_K)

        crc_jobs = []

        crc_L = max(a.L)

        for snr in a.snrs:
            seed = (
                a.seed
                + int(1000 * snr)
                + 999
            )

            crc_jobs.append(
                (
                    "crc",
                    crc_L,
                    snr,
                    a.N,
                    crc_K,
                    a.K,
                    a.blocks,
                    seed,
                )
            )

        print(
            f"Running {len(crc_jobs)} CRC simulation points "
            f"with {a.workers} worker(s)..."
        )

        rows.extend(
            run_jobs(
                crc_jobs,
                frozen_crc,
                a.workers,
            )
        )

    # ------------------------------------------------------------------
    # ML bound
    # ------------------------------------------------------------------

    if a.ml_bound:
        ml_jobs = []

        for snr in a.snrs:
            seed = (
                a.seed
                + int(1000 * snr)
                + 4242
            )

            ml_jobs.append(
                (
                    "ml-bound",
                    snr,
                    a.N,
                    a.K,
                    a.blocks,
                    32,
                    a.K,
                    seed,
                )
            )

        print(
            f"Running {len(ml_jobs)} ML-bound points "
            f"with {a.workers} worker(s)..."
        )

        rows.extend(
            run_jobs(
                ml_jobs,
                frozen,
                a.workers,
            )
        )

    # ------------------------------------------------------------------
    # Sort results into deterministic output order
    # ------------------------------------------------------------------

    decoder_order = {
        "sc": 0,
        "scl": 1,
        "ca-scl": 2,
        "ml-bound": 3,
    }

    rows.sort(
        key=lambda r: (
            decoder_order.get(r["decoder"], 99),
            r["L"],
            r["snr"],
        )
    )

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------

    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(
            [
                "decoder",
                "L",
                "EbN0_dB",
                "BER",
                "FER",
                "blocks",
            ]
        )

        for r in rows:
            w.writerow(
                [
                    r["decoder"],
                    r["L"],
                    r["snr"],
                    r["ber"],
                    r["fer"],
                    r["blocks"],
                ]
            )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    decoder_names = sorted(
        set(r["decoder"] for r in rows)
    )

    for decoder in decoder_names:
        L_values = sorted(
            set(
                r["L"]
                for r in rows
                if r["decoder"] == decoder
            )
        )

        for L in L_values:
            rr = [
                r
                for r in rows
                if r["decoder"] == decoder
                and r["L"] == L
            ]

            rr.sort(key=lambda r: r["snr"])

            ax.semilogy(
                [r["snr"] for r in rr],
                [r["fer"] for r in rr],
                "o-",
                label=f"{decoder}, L={L}",
            )

    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel("Word / frame error rate")
    ax.set_title(
        f"Polar N={a.N}, effective rate 1/2, "
        f"design={a.design_snr:g} dB"
    )
    ax.grid(True, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(a.out, dpi=180)

    print("saved", a.out)
    print("saved", a.csv)


if __name__ == "__main__":
    main()
    