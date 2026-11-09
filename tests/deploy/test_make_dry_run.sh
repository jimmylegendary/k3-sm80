#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

# K3 lane 01(기준)과 K3.1 lane 11 모두 같은 dry-run 계약을 만족해야 한다.
for lane_id in 01-tp8-pp4-256k-safe 11-k31-tp8-pp4-256k-safe; do
  lane="$repo/deploy/$lane_id"

  module_output="$(
    cd "$repo"
    python3 -m deploy.tooling.remote up --lane "$lane" --dry-run
  )"
  check_output "$module_output"

  make_output="$(make --no-print-directory -s -C "$lane" dry-run)"
  check_output "$make_output"

  # gate dry-run은 선행 gate마다 argv 한 줄을 내고 아무것도 실행하지 않는다.
  gate_lines="$(grep -c '^K3-GATE-DRY gate=' <<<"$make_output")"
  incomplete_lines="$(grep -c '^K3-GATE-INCOMPLETE ' <<<"$make_output")"
  case "$lane_id" in
    01-*)
      [[ "$gate_lines" -eq 3 ]]  # k08 + kda(k01,k02)
      [[ "$incomplete_lines" -eq 2 ]]
      ! grep -q 'k11_dsa_indexer_sm80\|k12_sparse_mla_sm80' <<<"$make_output"
      ;;
    11-*)
      [[ "$gate_lines" -eq 5 ]]  # k08 + kda(k01,k02) + k11 + k12
      [[ "$incomplete_lines" -eq 4 ]]
      for gate in k08 kda k11 k12; do
        grep -q "^K3-GATE-INCOMPLETE lane=$lane_id gate=$gate mode=cpu-dry-run" <<<"$make_output"
      done
      grep -q '^K3-GATE-DRY gate=k11 argv=.*k31-sm80-openai:a3c14efb-sm80-v1.*/work/kernel-kit/k11_dsa_indexer_sm80/harness.py.*--cpu-dry-run' <<<"$make_output"
      grep -q '^K3-GATE-DRY gate=k12 argv=.*k31-sm80-openai:a3c14efb-sm80-v1.*/work/kernel-kit/k12_sparse_mla_sm80/harness.py.*--cpu-dry-run' <<<"$make_output"
      ;;
  esac
  ! grep -q 'K3-GATE-PASS' <<<"$make_output"
done

printf 'dry-run PASS lanes=2 commands=4 mutation=0\n'
