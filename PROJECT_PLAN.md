# When Does Sparsity Actually Help? — Revised Plan

**EN.601.434 Randomized and Big Data Algorithms, Final Project**
Zhuoxuan Lyn · Chenyu Li · April 2026

> This document supersedes the March abstract (`摘要汇报.pdf`). The crossover-threshold study was too incremental; the revised plan makes two substantive contributions that together form a problem → explanation → solution arc.

---

## 1. Revised Research Question

The March abstract asked: *how large is the gap between the theoretical crossover ρ\* = 1 − ℓ/d and the empirical crossover?* In practice this gap is large and well known; a log(1/δ) Chernoff correction only partially closes it. Two deeper questions remain open:

**Q1 (Why the gap exists).** What is the *correct* effective density that governs the FD/SFD tradeoff, and does it vary over the course of the sketch?

**Q2 (What to do about it).** Can a streaming algorithm decide, per batch and online, whether to use exact SVD (FD) or simultaneous iteration (SFD) — and match the oracle that picks the best subroutine for each batch?

---

## 2. Contribution D — Sketch-Drift Effective Density (theoretical)

### Claim
The relevant density for deciding FD vs. SFD is not the input density ρ = nnz(A) / (nd), but the **effective density of the working buffer** M = [B; batch] at shrink time:

$$
\rho_{\text{eff}}(t) \;=\; \frac{\operatorname{nnz}(B_t) + \operatorname{nnz}(\text{batch}_t)}{(\ell + r_t)\,d}.
$$

Because B is dense after the first shrink, nnz(B_t) = ℓd, so

$$
\rho_{\text{eff}}(t) \;=\; \frac{\ell d + \operatorname{nnz}(\text{batch}_t)}{(\ell + r_t)\,d} \;\geq\; \frac{\ell}{\ell + r_t}.
$$

This is a **lower bound independent of ρ**. For the paper's nnz-based batching (r_t chosen so nnz(batch_t) ≈ ℓd), ρ_eff is bounded below by roughly ½. The sparse subroutine therefore never sees a truly sparse operand after the first shrink.

### Consequence
The corrected crossover predicts:

$$
\hat\rho_{\text{drift}} \;=\; \rho^\* \cdot \frac{\operatorname{nnz}(\text{batch}_t)}{\ell d + \operatorname{nnz}(\text{batch}_t)}
\;\ll\; \rho^\*.
$$

This is a **multiplicative shrinkage of ρ\*** that explains the empirically observed early crossover without invoking retry overhead. It is a purely combinatorial argument (no δ, no Chernoff), which we find cleaner than the log(1/δ) correction.

### Experiments for D
- Instrument SFD to log ρ_eff(t) per shrink; plot trajectory on synthetic and real data.
- Verify that the empirical crossover ρ ≈ ρ_eff(1) rather than ρ\*.
- Compare to the log(1/δ) correction — show ρ̂_drift fits better at fixed δ.

---

## 3. Contribution A — Adaptive FD/SFD (algorithmic)

### Algorithm
At each shrink step we have a buffer M of known shape and nnz. Rather than committing to one subroutine globally, choose per-batch:

```
def shrink(M, ell):
    c_exact = alpha * rows(M) * d * ell            # cost of FD
    c_sfd   = beta * nnz(M) * ell * n_iter + rows(M) * ell^2
    if c_sfd < c_exact:
        return simultaneous_iteration(M, ell)
    else:
        return exact_svd(M, ell)
```

The constants α, β are calibrated once on the running hardware (≈ 30 s of micro-benchmarks at startup). This is a **hardware-calibrated online selector**, not a static flop model.

### Guarantee (to prove)
Let OPT be the oracle that chooses the cheaper of {FD, SFD} per batch with perfect foresight. The adaptive algorithm achieves

$$
T_{\text{adaptive}} \;\leq\; (1 + \varepsilon)\,T_{\text{OPT}} + O(\text{calibration})
$$

where ε is the multiplicative error of the calibrated cost model. Because the decision is local and the subroutines are exchangeable (both return a valid rank-ℓ sketch satisfying the FD invariant), correctness is preserved — only speed is traded.

### Approximation bound
The worst-case covariance guarantee degrades to the weaker of the two: whenever a batch is handled by SFD, that batch contributes the (6/41)·(1/ℓ)‖·‖²_F term; FD batches contribute the optimal (1/ℓ)‖·‖²_F term. Net guarantee is

$$
\|A^\top A - B^\top B\|_2 \;\leq\; \frac{1}{\alpha_{\text{mix}}\,\ell - k}\,\|A - A_k\|_F^2
$$

with α_mix ∈ [6/41, 1] depending on the fraction of SFD-handled batches.

### Experiments for A
- **Homogeneous inputs**: on uniform-density synthetic data, Adaptive should match whichever of FD/SFD is faster. Confirms no overhead.
- **Mixed-density stream**: construct inputs with a density phase transition (first half ρ = 0.01, second half ρ = 0.3). Pure FD wastes work on the sparse half; pure SFD pays retry on the dense half. Adaptive should beat both.
- **Real data**: MovieLens 1M, 20 Newsgroups. Report Adaptive vs. FD vs. SFD wall-clock and covariance error.
- **Hardware sensitivity**: calibrate on CPU vs. GPU (Colab T4) — show that Adaptive automatically tracks the hardware-specific crossover while fixed SFD does not.

---

## 4. Revised Experimental Plan

| Setting | Dataset | Varying | Fixed |
|---|---|---|---|
| Synthetic homogeneous | Random sparse | ρ ∈ [0.001, 0.5] | n = 10⁴, d = 10³, ℓ = 20 |
| Synthetic mixed-density | Two-phase stream | phase boundary | ρ₁ = 0.01, ρ₂ = 0.3 |
| Real | MovieLens 1M | ℓ ∈ {10, 30, 100} | — |
| Real | 20 Newsgroups | ℓ ∈ {10, 30, 100} | — |
| Hardware | Synthetic | {CPU, GPU} | ρ = 0.01, ℓ = 30 |

For each run: wall-clock time, per-batch subroutine choice (for Adaptive), ρ_eff(t) trajectory, and ‖A⊤A − B⊤B‖₂ / ‖A − A_k‖²_F.

Each synthetic configuration is averaged over **5 seeds** (median ± IQR).

---

## 5. Fixes to Existing Code

Before running the new experiments, the following bugs in the current repo need fixing (see review notes):

1. `sfd.py`: change buffer filling to **nnz-based** (`while nnz(batch) < ell*d`), not row-based.
2. `sfd.py`: add real **VerifySpectral** — estimate ‖M⊤M − shrink_result⊤·shrink_result‖₂ via randomized power iteration; retry if above tolerance.
3. `benchmark.py`: replace `(1 − ℓ/d) / n_iter` "adjusted threshold" with ρ̂_drift from §2.
4. `benchmark.py`: add seed loop, report median + IQR.
5. `datasets.py`: upgrade to MovieLens 1M as originally planned.
6. Add `adaptive.py` implementing §3.
7. Add `calibrate.py` fitting α, β from short micro-benchmarks.

---

## 6. Deliverables

1. **Theoretical**: the ρ_eff bound (§2) and the competitive-ratio statement for Adaptive (§3), both with proofs.
2. **Implementation**: adaptive.py + fixed sfd.py, released on the existing GitHub repo.
3. **Empirical**: five-panel figure — (a) ρ_eff trajectory, (b) empirical vs. predicted crossover, (c) homogeneous-input speedup, (d) mixed-stream speedup, (e) CPU-vs-GPU crossover shift.
4. **Paper**: 6-page write-up. Structure — ρ\* is wrong → ρ_eff explains why → Adaptive is how to fix it.

---

## 7. Timeline (4 weeks)

| Week | Task |
|---|---|
| 1 | Fix sfd.py (nnz batching + VerifySpectral). Instrument ρ_eff logging. |
| 2 | Implement adaptive.py + calibrate.py. Prove ρ_eff lower bound and competitive ratio. |
| 3 | Run full experimental grid (synthetic + real + CPU/GPU). Generate figures. |
| 4 | Write up paper. Ablations and edge cases (low ℓ, highly skewed nnz). |
