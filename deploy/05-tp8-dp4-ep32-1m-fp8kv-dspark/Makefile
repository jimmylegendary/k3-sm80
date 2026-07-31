SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PYTHON ?= python3
LANE_ID := 05-tp8-dp4-ep32-1m-fp8kv-dspark
REQUIRED_GATES := k08 c01 k09 k10 ep 1m dspark
PREREQUISITE_GATES := k08 c01 k09
COMPOSE := docker compose --env-file .env
REPO_ROOT := $(abspath ../..)
REMOTE := PYTHONPATH=$(REPO_ROOT) $(PYTHON) -m deploy.tooling.remote
GATE := PYTHONPATH=$(REPO_ROOT) $(PYTHON) -m deploy.tooling.gates
RUN_ROOT := $(shell PYTHONPATH=$(REPO_ROOT) $(PYTHON) -c 'from pathlib import Path; from deploy.tooling.lane_schema import LaneEnv; e=LaneEnv.load(Path(".env")); print(Path(e.values["K3_SERVING_ROOT"])/"runs"/"$(LANE_ID)")')
INTEGRATION_DIR := $(RUN_ROOT)/integration

.PHONY: help config preflight dry-run up down status logs smoke gates \
	verify-gates verify-all-gates gate-k08 gate-kda gate-c01 gate-k09 \
	gate-load gate-context gate-dspark baseline calibrate

help:
	@printf '%s\n' \
	  '$(LANE_ID)' \
	  '  make config       Render Compose configuration' \
	  '  make preflight    Validate all four nodes and shared assets' \
	  '  make gates        Run standalone/communication prerequisites' \
	  '  make up/down      Start or stop this lane on four nodes' \
	  '  make gate-load    Validate checkpoint load/topology' \
	  '  make gate-context Validate this lane context target' \
	  '  make baseline     Save target-only DSpark comparison fixture' \
	  '  make gate-dspark  Validate DSpark against prior-lane fixture' \
	  '  make calibrate    Test current MAX_NUM_SEQS at full context' \
	  '  make smoke        Call OpenAI models/completions endpoints' \
	  '  make status/logs  Inspect this lane on all four nodes'

config:
	@$(COMPOSE) config
	@PYTHONPATH=$(REPO_ROOT) $(PYTHON) -m deploy.tooling.command \
	  --env-file .env --all-ranks

preflight:
	@$(PYTHON) scripts/preflight.py --env-file .env

dry-run:
	@$(REMOTE) up --lane $(CURDIR) --dry-run
	@$(GATE) run --lane $(CURDIR) --cpu-dry-run

gates: preflight
	@$(GATE) run --lane $(CURDIR)
	@$(GATE) verify-prerequisites --lane $(CURDIR)

verify-gates:
	@$(GATE) verify-prerequisites --lane $(CURDIR)

verify-all-gates:
	@$(GATE) verify-all --lane $(CURDIR)

gate-k08:
	@$(GATE) run --lane $(CURDIR) --gate k08

gate-kda:
	@$(GATE) run --lane $(CURDIR) --gate kda

gate-c01:
	@$(GATE) run --lane $(CURDIR) --gate c01

gate-k09:
	@$(GATE) run --lane $(CURDIR) --gate k09

up: preflight verify-gates
	@$(REMOTE) up --lane $(CURDIR)

down:
	@$(REMOTE) down --lane $(CURDIR)

status:
	@$(REMOTE) status --lane $(CURDIR)

logs:
	@$(REMOTE) logs --lane $(CURDIR)

smoke:
	@$(PYTHON) scripts/smoke.py --env-file .env --mode smoke \
	  --report $(INTEGRATION_DIR)/smoke.json

gate-load:
	@$(PYTHON) scripts/smoke.py --env-file .env --mode load \
	  --report $(INTEGRATION_DIR)/load.json
	@$(GATE) record --lane $(CURDIR) --gate ep \
	  --evidence $(INTEGRATION_DIR)/load.json

gate-context:
	@$(PYTHON) scripts/smoke.py --env-file .env --mode context \
	  --report $(INTEGRATION_DIR)/context.json
	@$(GATE) record --lane $(CURDIR) --gate 1m \
	  --evidence $(INTEGRATION_DIR)/context.json

baseline:
	@$(PYTHON) scripts/smoke.py --env-file .env --mode baseline \
	  --report $(INTEGRATION_DIR)/baseline.json

gate-dspark:
	@if [[ '1' != '1' ]]; then \
	  printf 'K3-DSPARK-NOT-APPLICABLE lane=%s\n' '$(LANE_ID)'; exit 2; \
	fi
	@$(PYTHON) scripts/smoke.py --env-file .env --mode dspark \
	  --report $(INTEGRATION_DIR)/dspark.json
	@$(GATE) record --lane $(CURDIR) --gate k10 \
	  --evidence $(INTEGRATION_DIR)/dspark.json
	@$(GATE) record --lane $(CURDIR) --gate dspark \
	  --evidence $(INTEGRATION_DIR)/dspark.json

calibrate:
	@if [[ '1' != '1' ]]; then \
	  printf 'K3-CALIBRATE-NOT-APPLICABLE lane=%s\n' '$(LANE_ID)'; exit 2; \
	fi
	@$(PYTHON) scripts/smoke.py --env-file .env --mode calibrate \
	  --report $(INTEGRATION_DIR)/calibrate.json
