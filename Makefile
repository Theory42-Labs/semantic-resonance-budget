.PHONY: setup run test lint fmt clean
setup:
	python -m pip install -U pip
	pip install -r requirements.txt
run:
	python experiments/phase2/runners/run_phase2.py --config experiments/phase2/config/defaults.yaml
test:
	pytest -q
lint:
	ruff check .
fmt:
	ruff check --select I --fix . && ruff format .
clean:
	rm -rf experiments/phase2/reports/*