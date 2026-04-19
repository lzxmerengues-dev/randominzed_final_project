#!/usr/bin/env bash
# run_all.sh — Reproduce every figure from scratch.
#
# Usage:
#   bash run_all.sh              # quick mode (default, ~10 min on a laptop)
#   bash run_all.sh --full       # PDF-scale, ~3-4 hours
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

FULL_FLAG=""
if [[ "${1:-}" == "--full" ]]; then
    FULL_FLAG="--full"
fi

echo "====== Sanity tests ======"
python -m pytest tests/ -q

echo "====== Experiment 1: density sweep ======"
python experiments/exp1_density_sweep.py ${FULL_FLAG}

echo "====== Experiment 2: rho_eff trajectory ======"
python experiments/exp2_rho_eff_trajectory.py

echo "====== Experiment 3: mixed-density stream ======"
python experiments/exp3_mixed_stream.py

echo "====== Experiment 4: real datasets ======"
python experiments/exp4_real_data.py

echo "====== Experiment 5: CPU vs GPU ======"
python experiments/exp5_cpu_vs_gpu.py

echo ""
echo "====== Done. Figures in figures/, raw results in results/. ======"
ls -la figures/ results/
