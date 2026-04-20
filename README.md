# When Does Sparsity Actually Help?

Empirical and theoretical analysis of Sparse Frequent Directions (Ghashami, Liberty, Phillips, KDD 2016) vs. classical Frequent Directions (Liberty 2013).

**EN.601.434 Randomized and Big Data Algorithms** — final project.

Authors: Zhuoxuan Lyn · Chenyu Li.

See `PROJECT_PLAN.md` for the research narrative and `IMPLEMENTATION.md` for implementation notes.

## Two contributions

- **Sketch-drift effective density (D).** After the first shrink the dense sketch itself has nnz = ℓd, so the effective density of `M = [B; batch]` is bounded below by ℓ/(ℓ + r_t) — independent of input ρ. This explains why the empirical FD/SFD crossover is far below the textbook ρ\* = 1 − ℓ/d.

- **Adaptive FD/SFD (A).** A per-batch selector using a hardware-calibrated cost model. Picks exact SVD or SimultaneousIteration per shrink. Correctness is preserved: both branches satisfy the FD invariant; the bound degrades to `1/(α_mix·ℓ − k)` with α_mix ∈ [6/41, 1].

## Install

```bash
git clone https://github.com/lzxmerengues-dev/randominzed_final_project
cd randominzed_final_project
pip install -r requirements.txt
```

Optional GPU support (Experiment 5): `pip install cupy-cuda12x`.

### Running Experiment 5 on Google Colab (no local GPU required)

If you do not have a CUDA GPU locally, the CPU-vs-GPU experiment can be reproduced on a free Colab T4:

1. Open https://colab.research.google.com and create a new notebook.
2. `Runtime` → `Change runtime type` → select `T4 GPU` → `Save`.
3. Run the following in a cell:

```python
!git clone https://github.com/lzxmerengues-dev/randominzed_final_project.git
%cd randominzed_final_project
!pip install -r requirements.txt
!pip install cupy-cuda12x
!python experiments/exp5_cpu_vs_gpu.py
```

Outputs land in `results/exp5.json` and `figures/exp5_cpu_vs_gpu.{pdf,png}`; download them from the left file panel.

## Datasets

The synthetic experiments (Exp 1, 2, 3, 5) run without any download.

Real datasets for Experiment 4:

```bash
# MovieLens 1M
curl -L https://files.grouplens.org/datasets/movielens/ml-1m.zip -o ml-1m.zip
unzip ml-1m.zip -d data/ && rm ml-1m.zip

# MovieLens 100K (optional fallback)
curl -L https://files.grouplens.org/datasets/movielens/ml-100k.zip -o ml-100k.zip
unzip ml-100k.zip -d data/ && rm ml-100k.zip

# 20 Newsgroups fetched automatically via scikit-learn.
```

## Run

Quick mode (default, ~10 min on a laptop):

```bash
bash run_all.sh
# or:  make quick
```

Full PDF-scale experiments (~3-4 hours):

```bash
bash run_all.sh --full
```

Individual experiments:

```bash
make exp1   # density sweep
make exp2   # rho_eff trajectory
make exp3   # mixed-density stream
make exp4   # real datasets
make exp5   # CPU vs GPU
```

Tests:

```bash
make test
```

## Outputs

- `results/expN.json` — raw measurements (reproducible from seed).
- `figures/expN_*.pdf` + `.png` — figures for the paper.
- `~/.cache/sfd_calibration/` — cached (α_fd, β_sfd) coefficients per `(platform, d, ℓ)`.

Every figure is regenerated only from the corresponding JSON; rerunning plotting is free once the data exists.

## Repo layout

```
.
├── fd.py                  Frequent Directions (2ℓ buffer)
├── sfd.py                 Sparse FD: nnz batching + BoostedSparseShrink
├── adaptive.py            Per-batch FD/SFD selector
├── calibrate.py           Hardware-aware cost-model calibration
├── metrics.py             covariance_error, relative_error, ρ_eff, summaries
├── datasets.py            Synthetic + MovieLens 100K/1M + 20 Newsgroups
├── benchmark.py           run_once / run_seeds helpers
├── experiments/
│   ├── exp1_density_sweep.py
│   ├── exp2_rho_eff_trajectory.py
│   ├── exp3_mixed_stream.py
│   ├── exp4_real_data.py
│   └── exp5_cpu_vs_gpu.py
├── tests/                 pytest unit tests for every module
├── figures/               generated
├── results/               generated (JSON)
├── run_all.sh             orchestrator
├── Makefile               convenience targets
├── PROJECT_PLAN.md        research plan
├── IMPLEMENTATION.md      implementation notes
└── requirements.txt
```

## Reproducibility checklist

- Every experiment runs ≥ 3 seeds and reports median ± IQR.
- Randomness is seeded via `numpy.random.default_rng(seed)` and `np.random.seed(seed)` — no implicit global state inside the algorithms.
- Figures are regenerated deterministically from `results/expN.json`.
- Calibration is cached but can be forced off (`calibrate(use_cache=False)` or `make clean-cache`).

## License

Code for academic use only. MovieLens and 20 Newsgroups have their own licenses.
