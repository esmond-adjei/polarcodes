# Polar codes: SC, SCL, and CRC-aided SCL

A reproducible Python reference implementation of Tal--Vardy successive-cancellation list decoding, with exact LLR updates, stable path metrics, CRC-aided selection, systematic encoding, and channel-dependent construction experiments.

## Reproduction target

The main target is Tal & Vardy, *List Decoding of Polar Codes*:

- BPSK over BI-AWGN
- `N = 2048`
- effective information rate `R = 1/2`
- design `Eb/N0 = 2 dB`
- list sizes `L = 1, 2, 4, 8, 16, 32`
- word-error-rate curves

The paper's CRC experiment uses 1040 unfrozen positions: the first 1024 carry payload and the last 16 carry CRC. Thus the effective information rate remains `1024/2048 = 1/2`.

The paper states that its code construction was performed using Tal & Vardy's channel-construction method. This repository provides `tv-mc`, a deterministic Monte-Carlo density-evolution approximation to that construction, plus a faster Gaussian-approximation constructor for comparison. The exact internal frozen-set sequence used for the published simulations is not provided by the paper, so bit-for-bit curve identity cannot be claimed without recovering that sequence or the authors' original simulation implementation.

## Layout

```text
src/polar/
  core.py          polar transform + Gaussian-approximation construction
  construction.py  Monte-Carlo density-evolution construction (TV-MC)
  channel_sc.py    BPSK/AWGN channel + SC primitives
  scl.py           SCL + CRC-aided SCL
  crc.py           CRC-16-CCITT
  systematic.py    systematic polar encoder
  sim.py           Monte-Carlo simulation + empirical ML lower bound
scripts/
  replicate_tal_vardy.py
 tests/
  test_basic.py
  test_channel.py
  test_reproduction.py
 docs/
  reproduction.md
  method.md
  tutorial.md
  source papers
```

## Setup

```bash
uv sync
uv run pytest -q
```

## Paper-scale run

Vectorized NumPy SCL (~0.02 s/block SC, ~0.04 s/block L=32 at N=2048;
~7 min per 10,000-block point single-core at L=32, ~3-5 min at smaller L).
The full 30-point sweep below is under an hour on an 8-core machine with
the parallel script; use `--workers` to spread independent (decoder, L, SNR)
points across cores:

```bash
uv run python scripts/replicate_tal_vardy_parallel.py \
  --N 2048 --K 1024 \
  --design-snr 2 \
  --snrs 1 1.5 2 2.5 3 \
  --blocks 10000 \
  --L 1 2 4 8 16 32 \
  --construction tv-mc \
  --crc --ml-bound \
  --workers 8
```

Start small to validate trends before committing compute (seconds to minutes):

```bash
uv run python scripts/replicate_tal_vardy_parallel.py \
  --N 256 --K 128 --design-snr 1 \
  --snrs 1 2 3 --blocks 500 --L 1 2 4 8 --workers 4
```

`10,000` blocks is only a starting point. At FER around `1e-5`, substantially more blocks are required to estimate the tail reliably. Increase the simulation budget or use an error-event stopping rule for publication-quality curves.

## What was fixed

- Exact LLR box-plus instead of min-sum.
- Stable path-metric updates using `logaddexp`.
- Explicit effective-rate handling for CRC experiments.
- Tal--Vardy-style CRC placement: 16 CRC bits in the final 16 unfrozen positions.
- CRC-aware final list selection with best-path fallback.
- Systematic polar encoding.
- Empirical ML lower-bound estimator matching the paper's procedure.
- Deterministic channel construction with an explicit construction seed and sample budget.
- Tests for exact LLR updates, CRC, systematic encoding, encoder/decoder behavior, AWGN calibration, plus SCL/CA-SCL list behavior (L=1 ≡ SC, PM monotonicity, CRC selection), tv-mc determinism, and sim-harness smoke coverage.

## Important reproduction caveat

`tv-mc` is a numerical approximation of Tal--Vardy density evolution, not a claim that it reproduces the authors' internal quantized-channel construction bit-for-bit. The paper does not publish the exact frozen-set sequence used for Figure 1. For a strict forensic reproduction, recover the original frozen set or original implementation and feed it into the decoder unchanged.

The decoder uses the exact LLR box-plus and stable path metrics with NumPy-vectorized segment recursion (bit-identical to the per-bit reference loop). It still copies path vectors explicitly rather than the paper's lazy-copy sharing, but copies run in C and paper-scale N=2048 sweeps are routine (see timings above).

## SNR convention

All SNR values are `Eb/N0` in dB for unit-energy BPSK:

```text
sigma^2 = 1 / (2 R Eb/N0)
LLR     = 2 y / sigma^2
```

For the CRC experiment, `R` is the **payload rate**, `1024/2048 = 1/2`, not the inner polar dimension `1040/2048`.
