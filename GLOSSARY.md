# Glossary

Terms from the papers and the code. Each row says what the thing is and
where it shows up in this repo.

## Channel setup

| Term | Meaning | In this repo |
|---|---|---|
| Block length ($N$) | Codeword size, always a power of two. Larger $N$ polarizes harder. | `--N` flag; every function takes $N$ |
| Rate ($R = K/N$) | Info bits per channel use. Enters the SNR math, so it is passed explicitly. | `rate` arg of `awgn_llr`, `snr_to_sigma` |
| BPSK | Bit 0 sent as $+1$, bit 1 as $-1$. | $s = 1 - 2x$ in `awgn_llr` |
| AWGN | Each received value is the symbol plus independent Gaussian noise of std $\sigma$. The only channel simulated. | `awgn_llr` in `channel_sc.py` |
| $E_b/N_0$ | Energy per info bit over noise density, in dB. X-axis of every curve. Differs from $E_s/N_0$ by $10 \log_{10}(R)$. | `snr_db` args; `snr_to_sigma` converts via $\sigma^2 = 1/(2 R \cdot E_b/N_0)$ |
| LLR | $\log(P(y\|0)/P(y\|1))$. Sign picks the bit, magnitude gives confidence. Exact value here: $L = 2y/\sigma^2$. | `awgn_llr` output; consumed by $f$, $g$, path metrics |
| Design SNR | $E_b/N_0$ the frozen set is optimized for. Match it to the operating point. | `design_snr_db` in `awgn_construction` |

## Polarization and construction

| Term | Meaning | In this repo |
|---|---|---|
| Channel polarization | Recursively combining channels drives each synthetic channel toward perfect or useless; the good fraction approaches capacity. | Arikan 2009 PDF in `docs/` |
| Synthetic (bit-) channel | One ordered position $u_0 \dots u_{N-1}$, each seeing a different effective channel. Position 0 is worst, $N-1$ best. | Decided in order by `sc_decode` |
| Frozen bit | Position fixed to $0$ on a bad channel. Boolean mask, True means frozen. | `frozen` arrays; `frozen_set_bec`, `awgn_construction` |
| Bhattacharyya ($Z$) | Error-probability bound in $[0,1]$; $0$ is perfect, $1$ is useless. Exact recursion under BEC: $Z^- = 2Z - Z^2$, $Z^+ = Z^2$. | `frozen_set_bec` in `core.py` |
| Gaussian approx. (GA) | Construction tracking only mean LLRs instead of full densities. | `awgn_construction` (Tal-Vardy 2013) |
| Phi function | $\phi(x) = E[\tanh(L/2)]$, maps mean LLR into $[0,1]$ so the check-node update runs in the probability domain. | `_phi` / `_phi_inv` in `core.py` |
| Generator matrix ($F^{\otimes n}$) | The polar transform, Kronecker power of $[[1,0],[1,1]]$. | `kron_power` (reference); `encode` (butterfly) |
| Bit-reversal ($B_N$) | Permutation in Arikan's $G_N = B_N F^{\otimes n}$. Absorbed into ordering here. | `bit_reverse` documents it; outside the data path |

## Decoding

| Term | Meaning | In this repo |
|---|---|---|
| SC decoding | Decide $u_0$, then $u_1$ given $u_0$, and so on. $O(N \log N)$ but one early error corrupts the rest. | `sc_decode` in `channel_sc.py` |
| $f$ node | Upper-branch update: LLR of the XOR of two bits, min-sum form $\mathrm{sign}(a)\mathrm{sign}(b) \min(\|a\|,\|b\|)$. | `f` in `channel_sc.py` |
| $g$ node | Lower-branch update $b + (1-2u)a$: adds observations when the upper bit is $0$, subtracts when $1$. | `g` in `channel_sc.py` |
| Partial sums | Upper-half decisions re-encoded through a half-size butterfly. What $g$ is conditioned on; raw decisions fail silently above $N=4$. | `_partial_sums` in `channel_sc.py` |
| SCL decoding | Keep $L$ paths through SC; each info bit forks survivors, best $L$ continue (Tal-Vardy Algs 16-18). $L=1$ is SC. | `scl_decode` in `scl.py` |
| Path metric | Per-path cost $\log(1 + \exp(-(1-2u)L))$ per bit; lower is likelier. | `_update_pm` in `scl.py` |
| CA-SCL | Return the best list path passing CRC-16; fall back to best path. CRC bits cost rate. | `ca_scl_decode` in `scl.py`; `crc.py` |

## Measurement

| Term | Meaning | In this repo |
|---|---|---|
| BER | Wrong message bits over total message bits; CRC bits excluded. | `sim.run_point` return value |
| FER | Fraction of blocks with any message-bit error. Above BER; the gap shows whether errors bunch up. | `sim.run_point` return value |
| Monte Carlo point | One (decoder, $E_b/N_0$) measurement over `n_blocks` seeded messages. Needs $\sim 10+$ observed errors to mean anything. | `run_point(N, K, frozen, snr_db, ...)` |
