PY = .venv/bin/python

# saved Bayesian fits in results/models are reused, delete them to refit (about an hour)
all: data baselines bayes simulate figures

setup:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt -e .

data:
	$(PY) scripts/download_data.py
	$(PY) scripts/01_prepare.py

baselines:
	$(PY) scripts/02_baselines.py
	$(PY) scripts/03_drivers.py

bayes:
	$(PY) scripts/04_hierarchical.py
	$(PY) scripts/05_rq1.py
	$(PY) scripts/08_robustness.py
	$(PY) scripts/10_sms_effect.py

simulate:
	$(PY) scripts/06_simulate.py
	$(PY) scripts/07_matching.py

figures:
	$(PY) scripts/09_figures.py

test:
	$(PY) -m pytest -q

.PHONY: all setup data baselines bayes simulate figures test
