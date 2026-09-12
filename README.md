# Polar Codes: SC → SCL → CA-SCL (Tal & Vardy 2012 replication)

Pedagogical codebase replicating **Tal & Vardy, "List Decoding of Polar Codes"
(ISIT 2011 / IEEE Trans. Inf. Theory 2015)**. Built up in successive stages:

| Stage | Module | Idea (paper ref.) |
|---|---|---|
| 0 Polar codes | `src/polar/core.py` | Arikan 2009: $F^{\otimes n}$ butterfly encoder, BEC Bhattacharyya + GA (Tal–Vardy 2013) construction |
| 1 Channel | `src/polar/channel_sc.py` | BPSK + AWGN, LLRs $L=2y/\sigma^2$ |
| 2 SC decoding | `src/polar/channel_sc.py` | Arikan SC with $f$ (min-sum) / $g$ + **partial sums** |
| 3 SCL decoding | `src/polar/scl.py` | Tal–Vardy list decoder, PM $\mathrel{+}=\log(1+e^{-(1-2u)L})$, keep best $L$ |
| 4 CA-SCL | `src/polar/scl.py` + `crc.py` | CRC-16 aided: pick best CRC-passing path |
| 5 Replication | `scripts/replicate_tal_vardy.py` | BER/FER vs SNR curves (paper Figs. 1–3 style) |

Docs: `docs/` holds Arikan 2009, Arikan tutorial, Tal–Vardy 2012 (list decoding),
Tal–Vardy 2013 (construction).

## Setup

```bash
uv sync
uv run pytest tests/ -q                       # unit checks (encode involution, noiseless round-trip)
uv run python scripts/replicate_tal_vardy.py --N 128 --K 64 \
    --snrs 1.0 2.0 3.0 --blocks 50 --L 4 --out results/curve128.png
```

Paper-scale run (slow in pure Python — start small, scale up):

```bash
uv run python scripts/replicate_tal_vardy.py --N 1024 --K 512 \
    --snrs 1.0 1.5 2.0 2.5 --blocks 200 --L 32 --out results/tal_vardy.png
```

## Key lessons encoded here

- SC's `g`-step needs **partial sums** (re-encoded upper decisions), not raw
  decisions — the classic implementation pitfall (see `channel_sc.py`).
- List decoding helps most at moderate SNR; CRC-aided SCL picks the correct
  path among the $L$ candidates (Tal–Vardy §IV).
- Construction (which $N-K$ channels to freeze) matters as much as decoding.

## Layout

```
src/polar/  core.py  channel_sc.py  scl.py  crc.py  sim.py
scripts/    replicate_tal_vardy.py
tests/      test_basic.py
docs/       the four papers (PDF)
results/    generated curves (git-ignored)
```
