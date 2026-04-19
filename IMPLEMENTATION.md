# Implementation Plan — SFD Crossover (A + D)

Based on the current GitHub repo (`lzxmerengues-dev/randominzed_final_project`, commit as of 2026-04-19). Companion to `PROJECT_PLAN.md`.

## 0. Target Repo Layout

```
randominzed_final_project/
├── fd.py                 # patched — 2ℓ buffer, doubling
├── sfd.py                # rewritten — nnz batching + real VerifySpectral
├── adaptive.py           # NEW — per-batch FD/SFD selector
├── calibrate.py          # NEW — hardware cost-model calibration
├── metrics.py            # NEW — covariance_error, relative_error, ρ_eff
├── datasets.py           # patched — MovieLens 1M loader + mixed-stream
├── benchmark.py          # refactored — orchestrator only
├── experiments/
│   ├── exp1_density_sweep.py
│   ├── exp2_rho_eff_trajectory.py        # D
│   ├── exp3_mixed_stream.py              # A
│   ├── exp4_real_data.py
│   └── exp5_cpu_vs_gpu.py                # A + D on GPU
├── figures/              # generated
├── results/              # JSON dumps, reproducible
├── tests/
│   ├── test_fd.py
│   ├── test_sfd.py
│   ├── test_adaptive.py
│   └── test_metrics.py
├── PROJECT_PLAN.md
├── IMPLEMENTATION.md
└── README.md
```

Everything in single-module files except experiments are split per figure. Each experiment dumps its raw results to `results/expN.json` so figures can be regenerated without re-running.

---

## 1. Milestone M1 — Fix FD and SFD (Week 1, days 1–3)

### 1.1 `fd.py` — patch

**Current bug** (fd.py:26-41): buffer size is ℓ, forcing SVD every ℓ rows. Liberty's original uses 2ℓ buffer with doubling — this halves the SVD count, which matters for fair comparison against SFD.

**Target signature — unchanged:**
```python
def frequent_directions(A, ell: int) -> np.ndarray: ...
```

**Changes:**
- Buffer shape `(2*ell, d)`, track `zero_row` up to `2*ell`.
- SVD fires when `zero_row == 2*ell`.
- After shrink, `zero_row = ell` (top ℓ rows hold the shrunk sketch; bottom ℓ rows are free).
- Keep the `sp.issparse → toarray` conversion — FD is supposed to be density-oblivious.

**Acceptance test** (`tests/test_fd.py`):
```python
def test_fd_covariance_guarantee():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 80))
    ell, k = 20, 5
    B = frequent_directions(A, ell)
    lhs = np.linalg.norm(A.T @ A - B.T @ B, 2)
    _, s, _ = np.linalg.svd(A, full_matrices=False)
    tail = float(np.sum(s[k:] ** 2))
    assert lhs <= tail / (ell - k) + 1e-6  # Liberty bound
```

### 1.2 `sfd.py` — rewrite

**Three concurrent fixes:**

**(a) nnz-based buffer filling.** Current (sfd.py:101-104) fills ℓ rows. Paper fills until `nnz(buffer) ≥ ℓd`. Implementation:

```python
def _collect_batch(A_csr, start: int, ell: int, d: int) -> tuple[sp.csr_matrix, int]:
    """Accumulate rows from A_csr[start:] until nnz reaches ell*d (or end)."""
    target = ell * d
    nnz_so_far, end = 0, start
    row_ptr = A_csr.indptr
    n = A_csr.shape[0]
    while end < n and nnz_so_far < target:
        nnz_so_far += row_ptr[end + 1] - row_ptr[end]
        end += 1
    return A_csr[start:end], end
```

**(b) Real `VerifySpectral` + `BoostedSparseShrink`.** Paper 6.1 — estimate `‖M⊤M − shrunk⊤·shrunk‖_2` via randomized power iteration, compare to tolerance, retry if above.

```python
def _verify_spectral(
    dense_sketch: np.ndarray,
    sparse_batch: sp.spmatrix,
    B_candidate: np.ndarray,
    tol: float,
    n_probes: int = 10,
) -> bool:
    """
    True iff ||M^T M - B^T B||_2 ≤ tol, estimated via block power iteration.
    M = [dense_sketch; sparse_batch] implicit — never materialized.
    """
    d = B_candidate.shape[1]
    Omega = np.random.standard_normal((d, n_probes))
    # y_M = M^T M @ Omega
    y_dense  = dense_sketch   @ Omega
    y_sparse = sparse_batch   @ Omega
    MtM_Om = dense_sketch.T @ y_dense + sparse_batch.T @ y_sparse
    BtB_Om = B_candidate.T @ (B_candidate @ Omega)
    diff = np.linalg.norm(MtM_Om - BtB_Om, 2)
    # Block power lower-bounds the true spectral norm; multiply by safety factor
    return diff <= tol * 2.0

def _boosted_shrink(dense_sketch, sparse_batch, ell, n_iter, delta_prob):
    """Retry _simultaneous_iteration_mixed until VerifySpectral passes."""
    max_retries = max(1, math.ceil(math.log(1.0 / max(delta_prob, 1e-10))))
    # Tolerance from paper: O(sigma_ell^2)
    for attempt in range(max_retries):
        s, Vt = _simultaneous_iteration_mixed(dense_sketch, sparse_batch, ell, n_iter)
        B_cand, zero_row = _shrink(s, Vt, ell)
        tol = float(s[ell-1]**2) if len(s) >= ell else 0.0
        if _verify_spectral(dense_sketch, sparse_batch, B_cand, tol):
            return B_cand, zero_row, attempt + 1
    return B_cand, zero_row, max_retries   # accept last attempt on budget exhaustion
```

Return the number of attempts — this is what the ρ̂(δ) theory predicts.

**(c) Cleaner `_simultaneous_iteration_mixed`.** The current one (sfd.py:7-49) is correct in spirit but does an extra QR that isn't needed. Replace with standard randomized SVD (Halko-Martinsson-Tropp):

```python
def _simultaneous_iteration_mixed(dense_sketch, sparse_batch, k, n_iter):
    d = dense_sketch.shape[1]
    k_eff = min(k, d)
    Omega = np.random.standard_normal((d, k_eff))
    Y = dense_sketch @ Omega + sparse_batch @ Omega   # no vstack needed for M^T M
    for _ in range(n_iter):
        # power step on M^T M = A_d^T A_d + A_s^T A_s
        Q, _ = np.linalg.qr(Y)
        Y = dense_sketch @ (dense_sketch.T @ Q) + sparse_batch @ (sparse_batch.T @ Q)
    Q, _ = np.linalg.qr(Y)
    # Project and small SVD
    B_small = np.vstack([dense_sketch @ Q, sparse_batch @ Q])   # (rows × k_eff)
    U, s, Wt = np.linalg.svd(B_small, full_matrices=False)
    Vt = Wt @ Q.T
    return s, Vt
```

The power iteration is on `M⊤M` which is d×d — never formed, always applied as `A_d⊤(A_d·)` + `A_s⊤(A_s·)`. Complexity: `O((sketch_rows·d + nnz(batch)) · k · n_iter)`.

**(d) Top-level loop.** Replace `sparse_frequent_directions` body with:

```python
def sparse_frequent_directions(A, ell, n_iter=4, delta_prob=0.1, instrument=False):
    A = sp.csr_matrix(A) if not sp.issparse(A) else A.tocsr()
    n, d = A.shape
    B = np.zeros((ell, d))
    sketch_rows, start = 0, 0
    log = [] if instrument else None
    while start < n:
        batch, end = _collect_batch(A, start, ell, d)
        dense_sketch = B[:sketch_rows]
        B, sketch_rows, attempts = _boosted_shrink(
            dense_sketch, batch, ell, n_iter, delta_prob
        )
        if log is not None:
            log.append(dict(
                t=len(log), start=start, end=end,
                batch_rows=batch.shape[0], batch_nnz=batch.nnz,
                sketch_rows=sketch_rows, attempts=attempts,
                rho_eff=(ell * d + batch.nnz) / ((ell + batch.shape[0]) * d),
            ))
        start = end
    return (B, log) if instrument else B
```

`instrument=True` returns per-shrink trajectory — this is **the raw data for Experiment 2 (ρ_eff)**.

**Acceptance test** (`tests/test_sfd.py`):
```python
def test_sfd_bound():
    rng = np.random.default_rng(0)
    A = sp.random(500, 80, density=0.05, random_state=rng, format="csr")
    ell, k = 20, 5
    B = sparse_frequent_directions(A, ell)
    A_dense = A.toarray()
    lhs = np.linalg.norm(A_dense.T @ A_dense - B.T @ B, 2)
    _, s, _ = np.linalg.svd(A_dense, full_matrices=False)
    tail = float(np.sum(s[k:] ** 2))
    alpha = 6.0 / 41.0
    assert lhs <= tail / (alpha * ell - k) + 1e-6  # SFD bound
```

---

## 2. Milestone M2 — Calibration and Adaptive (Week 2)

### 2.1 `calibrate.py` — new

Fit (α_fd, β_sfd) such that wall-time ≈ α_fd · (rows · d · ell) for FD and β_sfd · (nnz · k · n_iter + rows · ell²) for SFD. One-time, ~30 s on startup.

```python
def calibrate(d: int = 1000, ell: int = 20, n_iter: int = 4, seed: int = 0) -> dict:
    """
    Returns {'alpha_fd': float, 'beta_sfd': float, 'device': str}.
    Fits coefficients via small grid of (rows, density).
    """
    rng = np.random.default_rng(seed)
    grid = [(200, 0.5), (200, 0.05), (500, 0.5), (500, 0.05), (1000, 0.01)]
    fd_data, sfd_data = [], []
    for rows, rho in grid:
        A = sp.random(rows, d, density=rho, random_state=rng, format="csr")
        # FD on dense
        A_d = A.toarray()
        t = _time(lambda: _fd_one_shrink(A_d, ell))
        fd_data.append((rows * d * ell, t))
        # SFD on sparse
        t = _time(lambda: _sfd_one_shrink(A, ell, n_iter))
        sfd_data.append((A.nnz * ell * n_iter + rows * ell**2, t))
    alpha_fd  = _fit_slope(fd_data)
    beta_sfd  = _fit_slope(sfd_data)
    return {"alpha_fd": alpha_fd, "beta_sfd": beta_sfd, "device": "cpu"}
```

`_fit_slope` is simple least-squares through origin: `slope = sum(x·y) / sum(x²)`. No intercept — we care about scaling behavior, and hardware noise floors are captured downstream.

Cache result to `~/.cache/sfd_calibration.json` keyed by `(platform, d, ell)` so subsequent runs skip calibration.

### 2.2 `adaptive.py` — new

```python
def adaptive_frequent_directions(
    A, ell, calib: dict, n_iter: int = 4, delta_prob: float = 0.1,
    instrument: bool = False,
) -> np.ndarray | tuple:
    """
    Per-batch FD/SFD selector using hardware-calibrated cost model.

    Guarantee: correctness identical to FD/SFD (returns valid FD-invariant sketch).
    Cost: approximately min(T_FD, T_SFD) per batch, up to calibration error.
    """
    A = sp.csr_matrix(A) if not sp.issparse(A) else A.tocsr()
    n, d = A.shape
    alpha_fd, beta_sfd = calib["alpha_fd"], calib["beta_sfd"]
    B = np.zeros((ell, d)); sketch_rows, start = 0, 0
    log = [] if instrument else None
    while start < n:
        batch, end = _collect_batch(A, start, ell, d)  # same nnz-based
        rows_M = sketch_rows + batch.shape[0]
        nnz_M  = ell * d + batch.nnz   # dense sketch contributes ell*d
        c_fd  = alpha_fd * rows_M * d * ell
        c_sfd = beta_sfd * (nnz_M * ell * n_iter + rows_M * ell**2)
        if c_fd < c_sfd:
            # FD path: materialize and exact-SVD
            M = np.vstack([B[:sketch_rows], batch.toarray()])
            s, Vt = _exact_svd_shrink(M, ell)
            choice = "fd"; attempts = 1
        else:
            # SFD path: boosted shrink
            _, _, attempts = _boosted_shrink(B[:sketch_rows], batch, ell, n_iter, delta_prob)
            choice = "sfd"
        B, sketch_rows = _shrink(s, Vt, ell)
        if log is not None:
            log.append(dict(t=len(log), choice=choice, c_fd=c_fd, c_sfd=c_sfd,
                            attempts=attempts, rho_eff=nnz_M / (rows_M * d)))
        start = end
    return (B, log) if instrument else B
```

**Correctness claim**: because both subroutines return a valid rank-ℓ sketch satisfying the FD shrink invariant (`‖M‖²_F − ‖B‖²_F = ℓ · Δ`), the resulting sketch is valid regardless of per-batch choice. Bound degrades to `1 / (α_mix · ℓ − k)` with `α_mix ∈ [6/41, 1]`.

**Acceptance tests** (`tests/test_adaptive.py`):
1. On homogeneous low-density input, Adaptive picks SFD ≥ 80% of batches.
2. On homogeneous high-density input, Adaptive picks FD ≥ 80%.
3. Adaptive wall-time ≤ 1.1 × min(FD_time, SFD_time) on fixed seed.

---

## 3. Milestone M3 — Instrumentation, Datasets, Metrics (Week 2)

### 3.1 `metrics.py` — pull out of benchmark.py

```python
def covariance_error(A_dense, B): ...
def relative_error(A_dense, B, k: int): ...
def rho_eff(batch_nnz, ell, d, batch_rows): ...
def theoretical_flops_adaptive(log): ...   # sum over per-batch choices
```

Delete from `benchmark.py`; import from here.

### 3.2 `datasets.py` — patch

- Rename `load_movielens(100k)` → `load_movielens_100k`.
- Add `load_movielens_1m` targeting `ml-1m/ratings.dat` (delimiter `::`, 6040 × 3706, ρ ≈ 4.5%).
- Add:
  ```python
  def make_mixed_stream(n1, n2, d, rho1, rho2, seed=0) -> sp.csr_matrix:
      """Stream: n1 rows at density rho1, then n2 rows at density rho2."""
  ```
  For Experiment 3. The row *order* is what triggers Adaptive's switch.

- Add `make_synthetic_lowrank(n, d, rank, rho, noise=0.1, seed=0)` that generates a **true** approximate-low-rank matrix (current `make_synthetic` in `benchmark.py:65-80` breaks low-rank structure via elementwise mask — replace with row-sparse or column-block sparse pattern so rank survives).

### 3.3 `benchmark.py` — slim down

Reduce to:
- `run_once(A, ell, algo, seed)` → dict of metrics + log
- `run_seeds(A, ell, algo, n_seeds)` → dict of {median, IQR}
- CLI dispatcher that calls experiments.

No more plotting in this file — figures are per-experiment.

---

## 4. Milestone M4 — Experiments (Week 3)

Each experiment is a standalone script under `experiments/`. Each: (1) loads/generates data, (2) runs with ≥5 seeds, (3) dumps `results/expN.json`, (4) produces figure to `figures/expN.pdf`.

### 4.1 `exp1_density_sweep.py` — sanity check

Basically the existing experiment but correct:
- n = 10⁴, d = 10³, ℓ = 20.
- ρ ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5}.
- 5 seeds each.
- Compare FD, SFD, Adaptive wall-time; report median ± IQR.
- Plot with paper-ρ\*, drift-ρ̂, empirical crossover lines.

### 4.2 `exp2_rho_eff_trajectory.py` — contribution D evidence

- Run SFD with `instrument=True` on three inputs: ρ ∈ {0.005, 0.05, 0.2}.
- Plot ρ_eff(t) vs. t (shrink index) for each.
- Annotate the ℓ/(ℓ + r_t) lower bound.
- Annotate input ρ as dashed horizontal.
- Expected visualization: ρ_eff trajectories converge upward regardless of input ρ — the theoretical punchline.

### 4.3 `exp3_mixed_stream.py` — contribution A evidence

- Input: `make_mixed_stream(5000, 5000, 1000, 0.01, 0.3)` — sparse phase then dense.
- Run all three algorithms; record `log` from Adaptive.
- Figure: left = wall-time bar chart (FD/SFD/Adaptive), right = Adaptive's per-batch choice over time (stacked sparse/dense phases visible).
- Expected: Adaptive close to min at every phase; pure algorithms wasteful on the mismatched phase.

### 4.4 `exp4_real_data.py` — real-world validation

- MovieLens 1M, 20 Newsgroups.
- ℓ ∈ {10, 30, 100}.
- 5 seeds (only Adaptive and SFD are randomized; FD deterministic).
- Table output: wall-time, relative covariance error, Adaptive's FD/SFD mix ratio.

### 4.5 `exp5_cpu_vs_gpu.py` — hardware sensitivity

- Requires cupy (pip install cupy-cuda11x; run in Colab if no local GPU).
- Port FD and SFD to cupy (`cupyx.scipy.sparse`, `cupy.linalg.svd`).
- Re-run exp1 on GPU.
- Calibrate α_fd, β_sfd separately on GPU — expect α_fd·d·ℓ ≪ CPU (near-free dense GEMM) and β_sfd only slightly reduced (sparse matvec is memory-bound).
- Figure: two panels (CPU, GPU), each showing empirical ρ̂. Expected result: GPU ρ̂ ≪ CPU ρ̂, possibly near 0 (never use SFD on GPU).

---

## 5. Milestone M5 — Figures + Paper (Week 4)

Five paper figures correspond 1:1 to experiments. All using matplotlib with consistent style (`figures/style.mplstyle`).

Paper outline:
1. Intro, FD/SFD recap
2. §3 Sketch-drift effective density (theoretical)
3. §4 Adaptive FD/SFD (algorithmic)
4. §5 Experiments
5. §6 Hardware discussion (CPU vs GPU)
6. §7 Related work, conclusion

6 pages is enough; not trying to pad.

---

## 6. Concrete Task Ordering

Strict dependency order — each task unlocks the next:

| # | Task | File | Blocker for |
|---|------|------|-------------|
| 1 | Pull out metrics into `metrics.py` | metrics.py | everything |
| 2 | Patch `fd.py` 2ℓ buffer | fd.py | exp1 |
| 3 | Rewrite `sfd.py` (nnz batch + BoostedShrink) | sfd.py | exp1, exp2 |
| 4 | Write tests for FD/SFD | tests/ | merge confidence |
| 5 | Upgrade `datasets.py` (MovieLens 1M + mixed-stream + lowrank) | datasets.py | exp1, exp3, exp4 |
| 6 | Build `calibrate.py` | calibrate.py | adaptive.py |
| 7 | Build `adaptive.py` | adaptive.py | exp3, exp4, exp5 |
| 8 | Tests for adaptive | tests/ | merge |
| 9 | exp1 density sweep | experiments/ | figure 1 |
| 10 | exp2 ρ_eff trajectory | experiments/ | figure 2 |
| 11 | exp3 mixed stream | experiments/ | figure 3 |
| 12 | exp4 real data | experiments/ | figure 4 |
| 13 | GPU port + exp5 | experiments/ | figure 5 |
| 14 | Write paper | — | submission |

Tasks 1–5 in parallel with 6–7 (adaptive doesn't block FD/SFD fixes). Days 1–3 can do 1+2+3 in one session.

---

## 7. "Perfect" Acceptance Criteria

Before calling any milestone done:

- **All tests pass** (`pytest tests/`).
- **No random-seed flakiness**: every experiment script runs `numpy.random.default_rng(seed)` and `scipy.sparse.random(..., random_state=rng)` — no implicit global state.
- **Results are reproducible**: `python experiments/expN.py` regenerates `results/expN.json` bit-identical on the same seed.
- **Every plot traces back to raw data**: no figure is generated from in-memory arrays; always from `results/expN.json`.
- **Bounds hold in tests**, not just in the paper.
- **Calibration cache is portable**: explicit platform key, no reliance on cpuinfo quirks.
- **One-command reproduction**: `make all` (or `bash run_all.sh`) reproduces every figure from scratch in ≤ 4 hours on a laptop.

If any of the above fails, the milestone is not done — iterate.
