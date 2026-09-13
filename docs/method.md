# Method note: theory to code

How each piece of the papers maps to this repo, which conventions were
chosen, and where the code knowingly departs from Tal-Vardy. Read alongside
the PDFs in this folder.

## Encoding

`core.encode` implements the standard non-systematic polar transform
$x = u F^{\otimes n}$ as an in-place butterfly: at stage $s$ (step $2^s$),
$v[i+j] \mathrel{\oplus}= v[i+j+\mathrm{step}]$. Arikan writes the generator as $B_N F^{\otimes n}$ with an
explicit bit-reversal; that permutation is absorbed here into position
ordering, which capacity and decoder behavior do not depend on. One
check on this: `test_basic.py` asserts $\mathrm{encode}(\mathrm{encode}(u)) = u$, since the
transform is its own inverse.

## Construction (choosing frozen positions)

Two methods, both returning a boolean frozen mask with $N-K$ entries set.

BEC construction (`frozen_set_bec`) runs the exact Bhattacharyya recursion
$Z^- = 2Z - Z^2$, $Z^+ = Z^2$ from $Z = \epsilon$ and freezes the $N-K$ worst channels.
This is exact for the erasure channel and a good default everywhere.

AWGN construction (`awgn_construction`) follows the Gaussian approximation
of Tal-Vardy 2013 (and Chung-Richardson-Urbanke): track the mean LLR $m$ of
each synthetic channel from $m_0 = 2/\sigma^2$, with upper update
$\phi^{-1}(1-(1-\phi(m))^2)$ and lower update $2m$, then freeze the $N-K$ smallest
means. $\phi$ uses the Ha et al. piecewise approximation with bisection
inverse. The design SNR uses the same $E_b/N_0$ calibration as the channel
(see README), with rate $K/N$.

A note on what the GA buys here: at matched capacity the GA and BEC sets
largely agree (at $N=128$, $R=1/2$, GA at 1 dB design SNR shares all 64 info
positions with BEC at $\epsilon = 0.5$). The sets diverge as design SNR moves. If
you need paper-exact design-SNR curves, the $\phi$ approximation is the first
place to tighten.

## SC decoding

`sc_decode` recurses on contiguous minus/plus splits. Even positions of a
segment go through $f(a,b) = \mathrm{sign}(a) \mathrm{sign}(b) \min(|a|,|b|)$, the min-sum check
node; odd positions go through $g(a,b,u) = b + (1-2u)a$, the variable node
conditioned on the upper decision.

The subtle part is what "upper decision" means in $g$. It is not the raw
decoded bit. It is the partial sum: the upper-half decisions re-encoded
through a half-size butterfly. `_partial_sums` does this, and `sc_llr`
(the per-bit LLR used by SCL) shares the same path. Getting this wrong
still decodes $N=4$ correctly and fails silently above it, which is why the
noiseless round-trip test covers several block lengths.

## List decoding

`scl_decode` mirrors Tal-Vardy Alg 16-18 in LLR form. Each info bit forks
every surviving path in two; each frozen bit extends them in one. Path
metrics update by $\log(1 + \exp(-(1-2u)L))$ and only the $L$ best forks survive.
Candidates come back best-first, which is the Alg 19 selection. The
complexity is $O(L \cdot n^2)$ because paths hold full decision vectors and LLRs
are recomputed per bit; the paper's $O(L \cdot n \log n)$ comes from the lazy-copy
path sharing of Algs 8-13, which this repo skips for readability.

## CRC-aided selection

`ca_scl_decode` returns the best path whose info bits pass CRC-16-CCITT and
falls back to the best path otherwise. The CRC-fallback pattern follows
Niu-Chen rather than Tal-Vardy. One consequence for reading the curves:
with $K$ fixed, the CRC steals 16 info positions, so CA-SCL runs at a lower
effective rate than plain SCL and can trail it at short $N$. Compare at fixed
message length or at $N=2048$ to see the paper's ordering.

## Simulation harness

`sim.run_point` draws uniform messages, encodes, passes them through
`awgn_llr`, decodes, and counts bit and frame errors. Seeds are fixed per
call, so reruns are reproducible. Keep blocks high enough that the lowest
BER point sees at least tens of errors, or the comparison between decoders
is noise.

## Known limits

- Pure Python recursion: $N=1024$ with $L=32$ runs but slowly. Vectorizing the
  $f$/$g$ steps over paths is the obvious speedup.
- Min-sum $f$ instead of exact boxplus: small loss vs the paper's likelihood
  computations, standard in practice.
- No puncturing/shortening, no systematic encoding, no 5G NR sequence.
  The 3GPP reliability sequence would be a good cross-check for
  `awgn_construction` orderings.
