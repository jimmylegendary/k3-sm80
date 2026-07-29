#!/usr/bin/env bash
# 최신 결과 tgz 를 repo 에 커밋·푸시해 개발 호스트로 회신한다 (PLAN "반복 루프"의 회신 반쪽).
#
# 사용:   bash run/30-send-results.sh
# 결과:   results/incoming/ 에 최신 kernel-report tgz 커밋 + push.
#         개발 호스트(Claude)가 이걸 분석해 커널 수정(S3)을 진행한다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tgz=$(ls -t "$ROOT"/kernel-kit/k3-sm80-kernel-report-*.tgz 2>/dev/null | head -1) \
  || { echo "[send] 결과 tgz 없음 — 먼저 run/20-kernel-test.sh 를 실행" >&2; exit 3; }
mkdir -p "$ROOT/results/incoming"
cp "$tgz" "$ROOT/results/incoming/"
git -C "$ROOT" add results/incoming
git -C "$ROOT" commit -m "results: $(basename "$tgz") (사내 A100 실행 결과)"
git -C "$ROOT" push
echo "[send] push 완료: results/incoming/$(basename "$tgz")"
