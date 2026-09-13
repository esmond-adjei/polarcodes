# Reproduction protocol

## Target

Tal & Vardy, *List Decoding of Polar Codes*, Fig. 1 / Fig. 5:

- BPSK over BI-AWGN
- `N = 2048`
- effective information rate `R = 1/2`
- design `Eb/N0 = 2 dB`
- list sizes `L = 1, 2, 4, 8, 16, 32`
- word-error-rate curves

The CRC experiment follows the paper's construction: the inner polar code has
1040 unfrozen positions; 1024 carry payload and the last 16 carry a CRC, so the
effective transmitted information rate remains 1024/2048 = 1/2.

## Construction

The paper says its code construction was carried out using Tal & Vardy's
channel-construction method. The repository therefore provides `tv-mc`, a
Monte-Carlo density-evolution approximation that does not make the Gaussian
approximation used by the old implementation. It is deterministic under a
seed and its sample count is explicit.

This is not claimed to be bit-for-bit identical to an undocumented internal
simulation implementation. For a forensic reproduction, the exact frozen-set
sequence used by the authors would need to be recovered. `ga` remains available
for fast comparison.

## Decoder correctness

- exact LLR box-plus rather than min-sum
- numerically stable path metric via `logaddexp`
- CRC filtering only at final selection
- fallback to the best path when no candidate passes CRC
- explicit distinction between inner unfrozen dimension `K` and payload rate
- systematic encoder included for the paper's systematic experiment

## Monte Carlo

10,000 blocks per point is only a starting point. At a target FER of `1e-5`,
10,000 blocks cannot establish the tail reliably. Use enough blocks to collect
at least several dozen frame errors, or stop after a chosen error-event budget.
The raw CSV records the block count so the statistical uncertainty is visible.
