#!/usr/bin/env bash
# k01 IMA 원인(후보1: chunk.py h 오프셋 int32 오버플로) A/B 확정 + 수정 검증. GPU 필요.
#
# 이론(kernel-kit/vendor/K01-SM80-ANALYSIS.md): 단일 seq T>=87,424(청크 1366개)부터
# h 오프셋이 2^31 을 넘어 wrap → IMA. T=87,360 은 임계 직전이라 통과해야 한다.
#
# 사용:   bash run/26-k01-ab.sh [triton|full]     (기본 triton)
# 결과 판독 (마지막 AB-VERDICT 줄만 전달하면 됨):
#   FIXED        = A OK + B OK  → 진단·수정 모두 확정 (다음: run/20 으로 k01 전체 재검증)
#   REPRODUCED   = A OK + B FAIL → 진단 확정, 수정 미적용 상태 (git pull 확인)
#   UNEXPECTED   = 그 외        → 후보1 반증 또는 별개 문제 — 출력 전체 전달
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIER="${1:-triton}"
IMG="ghcr.io/jimmylegendary/k3-sm80-kernelkit:$TIER"
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "[ab] $IMG 없음 — 먼저: bash run/10-load-image.sh $TIER" >&2; exit 3; }

echo "[ab] 시작: tier=$TIER image=$IMG (A/B 진행상황은 즉시 출력)"
docker run --rm -i --gpus all -v "$ROOT":/work \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/.triton \
  "$IMG" python3 -u - <<'EOF'
import sys, traceback
sys.path.insert(0, "/work/kernel-kit")
sys.path.insert(0, "/work/kernel-kit/k01_kda_chunk")
import torch
import harness

kda_mod, src = harness.import_kda_ops()
fn = kda_mod.chunk_kda_with_fused_gate
print(f"AB import: {src}")

results = {}
for label, T in (("A", 87360), ("B", 87424)):
    tag = f"{label}(T={T})"
    try:
        inp = harness.build_inputs(torch, 1234, [T], "cuda")
        o, ht = harness.run_kernel(fn, inp, torch)
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(o.float()).all()) and bool(torch.isfinite(ht.float()).all())
        results[label] = "OK" if finite else "NONFINITE"
        print(f"AB {tag}: {results[label]}")
        del inp, o, ht
        torch.cuda.empty_cache()
    except Exception as e:
        results[label] = "FAIL"
        print(f"AB {tag}: FAIL {type(e).__name__}: {str(e)[:90]}")
        break   # IMA 후 CUDA 컨텍스트는 오염 — 이후 실행 무의미

a, b = results.get("A"), results.get("B")
if a == "OK" and b == "OK":
    print("AB-VERDICT: FIXED — 진단·수정 확정. 다음: bash run/20-kernel-test.sh <tier> k01_kda_chunk")
elif a == "OK" and b == "FAIL":
    print("AB-VERDICT: REPRODUCED — 진단 확정, 수정 미적용(git pull 후 재실행)")
else:
    print(f"AB-VERDICT: UNEXPECTED (A={a} B={b}) — 출력 전체를 전달")
EOF
