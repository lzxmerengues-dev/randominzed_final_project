.PHONY: all quick full test clean install

all: quick

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

quick:
	bash run_all.sh

full:
	bash run_all.sh --full

exp1:
	python experiments/exp1_density_sweep.py

exp2:
	python experiments/exp2_rho_eff_trajectory.py

exp3:
	python experiments/exp3_mixed_stream.py

exp4:
	python experiments/exp4_real_data.py

exp5:
	python experiments/exp5_cpu_vs_gpu.py

clean:
	rm -rf figures/*.pdf figures/*.png results/*.json
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete

clean-cache:
	rm -rf $${HOME}/.cache/sfd_calibration
