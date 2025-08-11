.PHONY: ensure_uv build_env train clean lint build run test stop package

UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## detecting if uv is available
ensure_uv:
	@[ -x "$(UV)" ] || (echo "uv not found, installing..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	@echo "Using uv at: $(UV)"

## Prepare environment
build_env: ensure_uv
	$(UV) sync

train: ensure_uv
	$(UV) run python train.py --train-data "data/train_data.csv" --train-labels "data/train_labels.csv"

clean: ensure_uv
	$(UV) run pyclean --verbose .
	rm -rf dist
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -f model/*.onnx
	rm -f model/*.log

lint: ensure_uv
	$(UV) run ruff format 
	$(UV) run ruff check

build:
	docker build -t inference_api .

run:
	docker run -d -p 8001:8000 --name inference_container inference_api
## docker run -d -p 8000:8000 --name inference_container inference_api

test: ensure_uv
	PYTHONPATH=. $(UV) run pytest -q

## Stop test server and remove container
stop:
	docker stop inference_container
	docker rm inference_container

## Package the problem for candidates
package:
	mkdir -p candidate_package
	mkdir -p candidate_package/inference
	mkdir -p candidate_package/tests
	cp train.py.template candidate_package/train.py
	cp pyproject.toml.template candidate_package/pyproject.toml
	cp inference/api.py.template candidate_package/inference/api.py
	cp Makefile candidate_package/Makefile
	cp Dockerfile candidate_package/Dockerfile
	cp README.md candidate_package/README.md
	cp -r tests/ candidate_package/
	zip -r candidate_package.zip candidate_package
	rm -rf candidate_package
