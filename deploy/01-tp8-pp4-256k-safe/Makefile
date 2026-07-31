SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PYTHON ?= python3
LANE_ID := 01-tp8-pp4-256k-safe
COMPOSE := docker compose --env-file .env
REPO_ROOT := $(abspath ../..)
REMOTE := PYTHONPATH=$(REPO_ROOT) $(PYTHON) -m deploy.tooling.remote

.PHONY: help config preflight dry-run up down status logs smoke gates calibrate

help:
	@printf '%s\n' \
	  '$(LANE_ID)' \
	  '  make config     Render Compose configuration' \
	  '  make preflight  Validate configuration and host prerequisites' \
	  '  make dry-run    Print four-node commands without executing them' \
	  '  make up/down    Start or stop this lane' \
	  '  make status     Show this lane on all four nodes' \
	  '  make logs       Tail this lane logs' \
	  '  make smoke      Call the OpenAI-compatible health/model endpoints' \
	  '  make gates      Run lane-specific prerequisite gates'

config:
	@$(COMPOSE) config

preflight:
	@$(PYTHON) scripts/preflight.py --env-file .env

dry-run:
	@$(REMOTE) up --lane $(CURDIR) --dry-run

up: preflight
	@$(REMOTE) up --lane $(CURDIR)

down:
	@$(REMOTE) down --lane $(CURDIR)

status:
	@$(REMOTE) status --lane $(CURDIR)

logs:
	@$(REMOTE) logs --lane $(CURDIR)

smoke:
	@$(PYTHON) scripts/smoke.py --env-file .env

gates:
	@printf 'K3-GATES-NOT-WIRED lane=%s\n' '$(LANE_ID)'
	@exit 2

calibrate:
	@printf 'K3-CALIBRATE-NOT-WIRED lane=%s\n' '$(LANE_ID)'
	@exit 2
