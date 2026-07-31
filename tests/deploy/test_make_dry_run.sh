#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
lane="$repo/deploy/01-tp8-pp4-256k-safe"

check_output() {
  local output="$1"
  [[ "$(grep -c '^K3-REMOTE-CMD profile=' <<<"$output")" -eq 4 ]]
  for profile in head worker1 worker2 worker3; do
    grep -q "^K3-REMOTE-CMD profile=$profile " <<<"$output"
  done
  ! grep -q 'ssh executed' <<<"$output"
  grep -q 'action=up' <<<"$output"
  grep -q 'node_rank=0' <<<"$output"
  grep -q 'node_rank=3' <<<"$output"
}

module_output="$(
  cd "$repo"
  python3 -m deploy.tooling.remote up --lane "$lane" --dry-run
)"
check_output "$module_output"

make_output="$(make --no-print-directory -s -C "$lane" dry-run)"
check_output "$make_output"

printf 'dry-run PASS commands=4 mutation=0\n'
