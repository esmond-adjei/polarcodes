# Polar codes: SC, SCL, and CRC-aided SCL

A readable Python replication of Tal and Vardy, "List Decoding of Polar
Codes" (2012/2015), built on Arikan's polarization (2009) and the Tal-Vardy
AWGN construction (2013). Each stage of the paper maps to one module, so you
can read the theory and the code side by side.

## What this replicates

Tal-Vardy showed that keeping L candidate paths through the successive
cancellation decoder, instead of one, closes most of the gap between SC
decoding and maximum-likelihood decoding. Their headline result: at N=2048,
R=1/2, an SCL decoder with L=32 plus a 16-bit CRC performs close to ML. This
repo reproduces that shape of result (BER/FER vs Eb/N0 curves for SC, SCL,
and CA-SCL) at sizes that run in reasonable time in pure Python.

## Repo layout

```
src/polar/
  core.py        polar transform, frozen-set construction (BEC + AWGN/GA)
  channel_sc.py  BPSK/AWGN channel LLRs, SC decoder, per-bit SC LLR helper
  scl.py         list decoder and CRC-aided variant
  crc.py         CRC-16-CCITT attach/check
  sim.py         BER/FER Monte Carlo harness
scripts/
  replicate_tal_vardy.py   paper-style curves (BER/FER vs Eb/N0)
tests/
  test_basic.py    encode properties, noiseless round-trip
  test_channel.py  Eb/N0 calibration vs Q-function theory
docs/
  method.md      theory-to-code map, conventions, known limits
  *.pdf          the four source papers
```

## Setup and quick start

```bash
uv sync
uv run pytest tests/ -q
```

A fast smoke run (N=128, L=4, about a minute):

```bash
uv run python scripts/replicate_tal_vardy.py --N 128 --K 64 \
    --snrs 1.0 2.0 3.0 --blocks 50 --L 4 --out results/curve128.png
```

A paper-scale run (N=1024+, L=32) uses the same flags. It is slow in pure
Python, so scale N, L, and --blocks gradually.

## SNR convention

Every SNR in this repo means Eb/N0 in dB for unit-energy BPSK. The noise
standard deviation follows sigma^2 = 1/(2*R*Eb/N0), and channel LLRs are
L = 2y/sigma^2. `tests/test_channel.py` pins this down by checking uncoded
BPSK against Q(sqrt(2*Eb/N0)): at 0 dB the measured BER is 0.0786, matching
theory. Before this calibration the code used sigma = 10^(-snr/20), which
coincides at R=1/2 but mislabels every other rate by 10*log10(2R) dB.

## How the decoders compare

At N=128, R=1/2 (50 blocks per point, GA construction at 1 dB design SNR):

| Eb/N0 | SC BER | SCL-4 BER | CA-SCL-4 BER |
|-------|--------|-----------|--------------|
| 1 dB  | 0.39   | 0.31      | 0.33         |
| 2 dB  | 0.37   | 0.21      | 0.24         |
| 3 dB  | 0.26   | 0.11      | 0.14         |

Two things to notice. List decoding roughly halves the error rate over SC at
each point, which is the paper's effect. And CA-SCL trails plain SCL here,
because the 16 CRC bits consume info positions at fixed K, so the comparison
is not rate-matched. At N=2048 the CRC gain dominates and the ordering flips
to match the paper. That crossover with block length is itself a result worth
reproducing.

## Paper-to-code map

| Paper algorithm | Code |
|---|---|
| SC main loop (Alg 1/2/5) | `sc_decode` in `channel_sc.py` |
| SCL main loop (Alg 16) | `scl_decode` in `scl.py` |
| continuePaths frozen/unfrozen (Alg 17/18) | fork, rank, prune to L in `scl_decode` |
| findMostProbablePath (Alg 19) | best-first ordering of returned paths |
| GA construction (Tal-Vardy 2013) | `awgn_construction` in `core.py` |

Deliberate simplifications, documented in `docs/method.md`: LLR and path
metrics replace the paper's likelihood domain, paths are full-vector copies
instead of the lazy-copy structure of Algs 8-13 (so SCL costs O(L*n^2), not
O(L*n log n)), and final CRC selection follows Niu-Chen rather than Tal-Vardy.

## Key implementation detail

The SC `g`-step needs partial sums, the re-encoded upper-half decisions, not
the raw decisions. Using raw decisions decodes N=4 correctly and fails
silently at larger N. `channel_sc.py` computes them with a half-size butterfly
(`_partial_sums`), and `sc_llr` shares the same path so SCL stays consistent.
This was caught by noiseless round-trip tests, which now cover N=16/64/256.
