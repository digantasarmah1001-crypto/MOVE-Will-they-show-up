PY = .venv/bin/python

# everything from raw data to the final tables. The Bayesian fits take about an hour;
# a saved fit in results/models is reused, delete it to refit
all: data baselines bayes simulate

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

simulate:
	$(PY) scripts/06_simulate.py
	$(PY) scripts/07_matching.py

test:
	$(PY) -m pytest -q

.PHONY: all setup data baselines bayes simulate test
