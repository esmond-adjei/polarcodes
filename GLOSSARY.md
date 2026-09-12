# Glossary

Terms from the papers and the code, in one place. Each entry says what the
thing is and where it shows up in this repo.

## Code and channel setup

**Block length (N).** Codeword size, always a power of two here (128, 1024,
2048). Larger N means stronger polarization: good channels get better, bad
ones get worse. Every script takes `--N`.

**Rate (R = K/N).** Fraction of positions carrying information. R=1/2 sends
one info bit per two channel uses. Rate enters the SNR math (see Eb/N0),
so it is passed explicitly to `awgn_llr` and `awgn_construction`.

**BPSK.** Binary phase-shift keying: bit 0 sent as +1, bit 1 as -1.
`awgn_llr` maps bits with `s = 1 - 2x`.

**AWGN.** Additive white Gaussian noise: each received value is the sent
symbol plus independent Gaussian noise of std sigma. The only channel this
repo simulates.

**Eb/N0.** Energy per info bit over noise spectral density, in dB. The
x-axis of every curve. Distinct from Es/N0 (energy per channel symbol):
Eb/N0 = Es/N0 / R. Mixing them up shifts curves by 10log10(R), about 3 dB
at R=1/2. `snr_to_sigma` converts with sigma^2 = 1/(2 R Eb/N0).

**LLR (log-likelihood ratio).** L = log(P(y|0)/P(y|1)). Positive favors 0,
magnitude measures confidence. For BPSK/AWGN the exact value is L = 2y/sigma^2.
Everything downstream (f, g, path metrics) consumes LLRs, never raw samples.

**Design SNR.** The Eb/N0 that `awgn_construction` optimizes the frozen set
for. Best practice is to match it to the operating point: a set designed at
0 dB wastes capacity at 4 dB and vice versa.

## Polarization and construction

**Channel polarization (Arikan 2009).** Recursively combining N copies of a
channel into N synthetic bit-channels drives each toward either perfect or
useless as N grows. The fraction of good ones approaches the channel
capacity, which is why polar codes achieve capacity.

**Synthetic (bit-) channel.** One of the N ordered positions u_0..u_{N-1},
each seeing a different effective channel after polarization. Position 0
sees the worst, position N-1 the best. Construction ranks them; decoding
decides them in order.

**Frozen bit.** A position fixed to 0 and known to the decoder, placed on a
bad synthetic channel. `frozen` masks in the code are boolean arrays, True
means frozen. Info bits go on the K best channels.

**Bhattacharyya parameter (Z).** A number in [0,1] bounding the error
probability of a bit-channel: 0 means perfect, 1 means useless. Under the
BEC it evolves exactly (Z- = 2Z - Z^2, Z+ = Z^2), which is what
`frozen_set_bec` propagates.

**Gaussian approximation (GA).** Construction method for AWGN that tracks
only the mean LLR of each synthetic channel instead of full densities
(Chung-Richardson-Urbanke; Tal-Vardy 2013). Implemented in
`awgn_construction`.

**Phi function.** phi(x) = E[tanh(L/2)] for L ~ N(x, 2x). Maps a mean LLR
into [0,1] so the check-node update can be computed in the probability
domain and mapped back with phi^{-1}. `_phi` / `_phi_inv` in `core.py`.

**Generator matrix (F^{⊗n}).** The polar transform, n-fold Kronecker power
of [[1,0],[1,1]]. `kron_power` builds it as a matrix (reference only);
`encode` applies the same transform as a butterfly without building it.

**Bit-reversal (B_N).** The permutation Arikan includes in G_N = B_N F^{⊗n}.
Absorbed into position ordering here, so `bit_reverse` documents the
convention but sits outside the data path.

## Decoding

**SC (successive cancellation) decoding.** Decide u_0, then u_1 given u_0,
and so on, each by its LLR sign. Cheap (O(N log N)) but brittle: one wrong
early decision corrupts the rest. `sc_decode`.

**f node (check node).** Upper-branch update combining two LLRs into the LLR
of their XOR: sign(a)sign(b) min(|a|,|b|) in the min-sum form used here.
Exact boxplus adds a small log correction term.

**g node (variable node).** Lower-branch update: b + (1-2u)a, where u is the
already-decided upper bit. Adds the two observations when they agree,
subtracts when they disagree.

**Partial sums.** The upper-segment decisions re-encoded through a
half-size butterfly. The g-step is conditioned on these, not on raw
decisions. `_partial_sums` computes them; using raw decisions instead is
the silent-at-N=4 bug documented in `docs/method.md`.

**SCL (list) decoding.** Keep L candidate paths through SC instead of one.
Each info bit forks every survivor; the 2L forks are ranked by path metric
and pruned to L (Tal-Vardy Algs 16-18). L=1 is SC; large L approaches
maximum likelihood. `scl_decode`.

**Path metric.** Per-path cost accumulated as log(1 + exp(-(1-2u)L)) per
decided bit: near zero when the decision agrees with its LLR, about |L|
when it contradicts it. Lower is likelier. `_update_pm`.

**CA-SCL (CRC-aided SCL).** Reserve 16 info bits for a CRC, list-decode,
then return the best path that passes the check (`ca_scl_decode`). The CRC
finds the right path when it is in the list but not ranked first. CRC bits
cost rate, so compare against SCL at matched message length.

## Measurement

**BER.** Bit error rate: wrong message bits over total message bits. CRC
bits are excluded from the count in `sim.py`.

**FER (frame/block error rate).** Fraction of blocks with any message-bit
error. Always above BER; the gap between them says whether errors arrive
alone or in bursts.

**Monte Carlo point.** One (decoder, Eb/N0) measurement over n_blocks random
messages. `run_point` seeds its RNG, so a point replays exactly. A BER
estimate from fewer than ~10 observed errors is noise; raise blocks until
the lowest point qualifies.
