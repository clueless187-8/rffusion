# rffusion — developer entry points. `make help` lists targets.
#
# `make check` is the definition of done from CLAUDE.md, encoded:
#   ruff check clean, pytest -q green, and the three synthetic-source
#   smoke commands (survey / analytics / fusion) run to completion on CPU.
# Everything here runs on x86 with no CUDA, no SDR, no Kismet, no camera.

PYTHON    ?= python3
PIP_FLAGS ?= --break-system-packages
SMOKE_DIR ?= .smoke
SRC       := rffusion tests

.DEFAULT_GOAL := help
.PHONY: help install check lint fmt fmt-check test smoke smoke-survey smoke-analytics smoke-fusion dev clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install package + dev/analytics extras (CPU path, no CUDA)
	$(PYTHON) -m pip install $(PIP_FLAGS) -e ".[dev,analytics]"

check: install lint test smoke ## Definition of done: install, ruff check, pytest, synthetic smokes
	@echo "make check: PASS"

lint: ## ruff check (must stay clean)
	ruff check $(SRC)

fmt: ## ruff format + autofix (writes files)
	ruff format $(SRC)
	ruff check --fix $(SRC)

fmt-check: ## ruff format --check (advisory; not part of `check` until the tree is formatted once)
	ruff format --check $(SRC)

test: ## pytest -q (must stay green)
	pytest -q

smoke: smoke-survey smoke-analytics smoke-fusion ## All three synthetic-source smokes

smoke-survey: ## survey smoke — synthetic SDR, 2 s, logs to $(SMOKE_DIR)
	mkdir -p $(SMOKE_DIR)
	$(PYTHON) -m rffusion.survey --seconds 2 -v --out $(SMOKE_DIR)

smoke-analytics: ## analytics smoke — synthetic Kismet feed, 500 events
	$(PYTHON) -m rffusion.analytics --events 500 -v

smoke-fusion: ## fusion smoke — correlation demo on synthetic stream
	$(PYTHON) -m rffusion.fusion --demo -v

dev: ## Run the survey build locally against the synthetic SDR until Ctrl-C
	$(PYTHON) -m rffusion.survey -v --out $(SMOKE_DIR)

clean: ## Remove caches and smoke output
	rm -rf .pytest_cache .ruff_cache $(SMOKE_DIR) build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
