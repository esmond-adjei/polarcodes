# Glossary

Terms from Arikan (2009) and Tal-Vardy (2012, 2013) as used in this codebase.
Each entry states what the object is, its mathematical definition, why it
matters for polar codes, and a concrete example.

## Channel setup

| Term | Definition |
|---|---|
| Block length ($N$) | Codeword size, constrained to $N = 2^n$ (e.g. $128$, $1024$, $2048$). Each doubling adds one polarization stage. Larger $N$ sharpens polarization: the set of good synthetic channels grows toward capacity, but decoding cost grows as $O(N \log N)$. |
| Rate ($R = K/N$) | Ratio of $K$ information-plus-CRC positions to $N$ transmitted symbols. $R = 1/2$ means one info bit per two channel uses. $R$ links $E_b/N_0$ (per info bit) to $E_s/N_0$ (per symbol) via $E_s/N_0 = R \cdot E_b/N_0$, so every noise-variance calculation must include $R$. |
| BPSK | Binary phase-shift keying with unit energy: bit $0 \mapsto +1$, bit $1 \mapsto -1$, i.e. $s = 1 - 2x$. The decoder receives $y = s + n$ where $n \sim \mathcal{N}(0, \sigma^2)$. No other modulation is used here. |
| AWGN | Additive white Gaussian noise channel: each received sample is $y = s + n$ with $n$ independent Gaussian noise of standard deviation $\sigma$. The channel is memoryless and symmetric; its LLR has the closed form $L = 2y/\sigma^2$. |
| $E_b/N_0$ | Energy per information bit over noise spectral density, expressed in dB. The $x$-axis of every BER/FER plot. Related to the per-symbol SNR by $E_b/N_0 = E_s/N_0 / R$, so confusing the two shifts curves by $10 \log_{10}(R)$ (about $-3$ dB at $R=1/2$). |
| LLR | Log-likelihood ratio $L = \log(P(y \mid 0)/P(y \mid 1))$. Sign decides the bit ($\ge 0$ favors $0$), magnitude measures confidence ($\|L\|=10$ is far more certain than $\|L\|=0.5$). For BPSK/AWGN, $L = 2y/\sigma^2$ exactly; all decoders consume LLRs, never raw $y$. |
| Design SNR | The $E_b/N_0$ value that the AWGN construction assumes when ranking synthetic channels. A set designed at $0$ dB puts info on channels reliable at low SNR; a set designed at $3$ dB trusts channels that only become reliable at higher SNR. Best practice is to match design SNR to the operating SNR. |

## Polarization and construction

| Term | Definition |
|---|---|
| Channel polarization | Arikan's recursive combining of $N$ copies of a channel $W$ into $N$ synthetic bit-channels $W_N^{(i)}$ such that, as $N$ grows, each $W_N^{(i)}$ tends to either perfect ($Z \to 0$) or useless ($Z \to 1$). The fraction of perfect channels approaches the symmetric capacity $I(W)$, which is why polar codes are capacity-achieving. |
| Synthetic (bit-) channel | The $i$-th effective channel seen by input bit $u_i$ after polarization, denoted $W_N^{(i)}$, with $0 \le i < N$. Ordered by construction reliability: $i=0$ is typically the worst, $i=N-1$ the best (exact order depends on $N$ and construction). The decoder decides $u_0, u_1, \dots, u_{N-1}$ in this order. |
| Frozen bit | An input position $u_i$ fixed to $0$ and known to encoder and decoder, placed on a bad synthetic channel. The boolean array `frozen` marks these: `True` means frozen. Information and CRC bits occupy the $K$ unfrozen (most reliable) positions. Example: at $N=8$, $K=4$, a typical frozen set is $\{0,1,2,4\}$. |
| Bhattacharyya parameter ($Z$) | An upper bound on the ML error probability of a bit-channel, $0 \le Z \le 1$ where $Z=0$ is perfect and $Z=1$ is useless. Under the binary erasure channel with erasure probability $\epsilon$, the recursion is exact: $Z^- = 2Z - Z^2$ (upper/worse branch) and $Z^+ = Z^2$ (lower/better branch), starting from $Z = \epsilon$. Becomes a ranking score: freeze the largest $Z$. |
| Gaussian approximation (GA) | Construction method for AWGN that tracks only the mean $m$ of each synthetic channel's LLR distribution (assumed Gaussian $\mathcal{N}(m, 2m)$) instead of full density evolution. Upper branch: $\phi^{-1}(1-(1-\phi(m))^2)$; lower branch: $2m$; starting from $m_0 = 2/\sigma^2$. Cheaper than density evolution and accurate enough for code design (Tal-Vardy 2013). |
| $\phi$ function | $\phi(x) = \mathbb{E}[\tanh(L/2)]$ for $L \sim \mathcal{N}(x, 2x)$, mapping a mean LLR $x \ge 0$ into $[0,1]$ with $\phi(0)=1$ and $\phi(\infty)=0$. Needed because the check-node (upper) update has no closed Gaussian form, so it is converted to the erasure-like domain via $\phi$, combined, and mapped back with $\phi^{-1}$. Approximated piecewise by Ha et al. |
| Generator matrix ($F^{\otimes n}$) | The polar transform matrix, the $n$-fold Kronecker power of $F = \begin{bmatrix}1 & 0 \\ 1 & 1\end{bmatrix}$. Encoding is $x = u \cdot F^{\otimes n}$ over $\mathrm{GF}(2)$. For $N=4$, $F^{\otimes 2} = \begin{bmatrix}1&0&0&0\\1&1&0&0\\1&0&1&0\\1&1&1&1\end{bmatrix}$, so $x = [u_0 \oplus u_1 \oplus u_2 \oplus u_3,\; u_1 \oplus u_3,\; u_2 \oplus u_3,\; u_3]$. The butterfly in `encode` applies this without building the matrix. |
| Bit-reversal ($B_N$) | The permutation that reverses the $n$-bit binary representation of an index (e.g. $011 \mapsto 110$). Arikan's full generator is $G_N = B_N F^{\otimes n}$. This codebase absorbs $B_N$ into position ordering, so the encoder and SC decoder agree on the same stage order without an explicit permutation step. |

## Decoding

| Term | Definition |
|---|---|
| SC (successive cancellation) decoding | Sequential decoder that decides $u_0$, then $u_1$ given $u_0$, up to $u_{N-1}$ given all previous decisions, each by the sign of its conditional LLR. Frozen bits are forced to $0$. Cost $O(N \log N)$, but fragile: one wrong early decision propagates because later LLRs are conditioned on it. |
| $f$ node | Upper-branch (check-node) LLR update combining two LLRs $a, b$ into the LLR of $u_{\mathrm{upper}} = u_0 \oplus u_1$. Exact form is $\log\frac{1+e^{a+b}}{e^{a}+e^{b}}$ (box-plus, implemented with `logaddexp`); min-sum approximation $f(a,b) = \mathrm{sign}(a)\mathrm{sign}(b)\min(|a|,|b|)$ drops the $\log(1+\cdot)$ correction at negligible loss. This codebase uses the exact form. |
| $g$ node | Lower-branch (variable-node) LLR update $g(a,b,u) = b + (1-2u) \cdot a$, where $u$ is the already-decided upper bit's partial sum. If $u=0$, the two observations reinforce ($b+a$); if $u=1$, they cancel ($b-a$). |
| Partial sums | Upper-half decisions re-encoded through a half-size polar butterfly, producing the codeword bits of that segment. The $g$-node is conditioned on these partial sums, not on raw decisions. Using raw decisions decodes $N=4$ correctly but fails silently at larger $N$ (e.g. a single-bit error on $u_3$ of $N=16$ maps to the wrong codeword half). |
| SCL (successive cancellation list) decoding | Generalization of SC that keeps $L$ candidate paths. Each information bit forks every survivor into two ($u=0$ and $u=1$); frozen bits extend without forking. All $2L$ forks are scored by path metric and pruned to the $L$ best (Tal-Vardy Algs 16–18). $L=1$ reduces to SC; $L \to \infty$ approaches ML. Complexity is $O(L \cdot N \log N)$ LLR work vectorized in NumPy (explicit path-vector copies in C on forks; bit-identical to the per-bit reference loop). |
| Path metric | Cumulative cost of a path's decisions: $\mathrm{PM} \mathrel{+}= \log(1 + \exp(-(1-2u)\cdot L))$ per bit, where $L$ is that bit's LLR. A decision agreeing with $L$ adds $\approx 0$; contradicting $L$ adds $\approx |L|$. Lower $\mathrm{PM}$ means more likely. Frozen bits still incur the $u=0$ penalty so paths stay comparable. |
| CA-SCL (CRC-aided SCL) | SCL with 16 CRC bits appended to the message: after list decoding, return the most likely path whose information bits pass CRC-16-CCITT, falling back to the overall best path if none pass. The CRC disambiguates when the true codeword is in the list but not ranked first. At fixed $K$, the 16 CRC bits reduce payload to $K-16$, so short-block comparisons must be rate-matched to be fair. |

## Measurement

| Term | Definition |
|---|---|
| BER (bit error rate) | Fraction of wrong message bits: $\mathrm{BER} = (\text{bit errors}) / (\text{total message bits})$. CRC bits are excluded from the count. Example: $100$ bit errors over $10{,}000$ message bits is $\mathrm{BER}=10^{-2}$. |
| FER (frame/word error rate) | Fraction of blocks containing at least one message-bit error: $\mathrm{FER} = (\text{bad blocks}) / (\text{total blocks})$. Always $\mathrm{FER} \ge \mathrm{BER}$ at the message level; the ratio indicates burstiness (many errors clustered in few blocks vs spread out). |
| Monte Carlo point | A single $(\text{decoder}, E_b/N_0)$ measurement estimated over `n_blocks` independently drawn random messages with a seeded RNG. Example: `run_point(N=128, K=64, snr_db=2.0, n_blocks=200)` draws 200 blocks. Reliable BER estimates need on the order of $10$ or more observed error events; below that the estimate is dominated by sampling noise. |
