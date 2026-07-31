#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MASTER_ADDR:?set MASTER_ADDR to the head-node service IP}"
: "${MASTER_PORT:?set MASTER_PORT to an unused shared port, e.g. 29610}"
: "${NODE_RANK:?set NODE_RANK to 0, 1, 2, or 3}"
: "${NNODES:=4}"

if [[ "$NNODES" != "4" ]]; then
  printf 'C01-LAUNCH-FAIL NNODES must be 4, got %s\n' "$NNODES" >&2
  exit 2
fi
if [[ ! "$NODE_RANK" =~ ^[0-3]$ ]]; then
  printf 'C01-LAUNCH-FAIL NODE_RANK must be 0..3, got %s\n' "$NODE_RANK" >&2
  exit 2
fi

export NNODES NODE_RANK
exec torchrun \
  --nnodes=4 \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  "$script_dir/harness.py" \
  --backend allgather_reducescatter \
  "$@"
