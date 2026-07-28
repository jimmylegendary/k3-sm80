"""S2 kernel-kit 사전조건 검사 — GPU/드라이버/torch/triton/compute-sanitizer + SM80 확인.

계약(docs/PLAN.md §S2 + 절대 제약):
  - GPU 모드: A100(SM80, capability (8,0)) + torch CUDA + triton 이 전부 있어야 진행.
    미충족 시 명확한 사유를 출력하고 exit 2 (= "skip") — 상위(run_all.sh)가 그대로 중단.
  - --cpu-dry-run: GPU 항목은 SKIP으로 표시만 하고 exit 0 (참조구현·판정로직·리포트 경로 검증용).
  - torch/triton import는 전부 가드 — GPU 없는 저작 호스트에서도 이 파일은 항상 실행 가능.

CLI:
  python3 env_check.py [--json] [--cpu-dry-run] [--quiet]
exit: 0 = 진행 가능 / 2 = GPU 사전조건 미충족(skip) / 1 = 내부 오류
"""

from __future__ import annotations

import glob
import json
import os
import platform
import shutil
import subprocess
import sys

EXIT_OK = 0
EXIT_ERR = 1
EXIT_PRECOND = 2

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _try(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def find_compute_sanitizer() -> str | None:
    """compute-sanitizer 실행파일 탐색: PATH → CUDA_HOME → /usr/local/cuda*."""
    p = shutil.which("compute-sanitizer")
    if p:
        return p
    cands = []
    if os.environ.get("CUDA_HOME"):
        cands.append(os.path.join(os.environ["CUDA_HOME"], "bin", "compute-sanitizer"))
    cands += sorted(glob.glob("/usr/local/cuda*/bin/compute-sanitizer"), reverse=True)
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def collect_env() -> dict:
    """리포트 env 블록 — 판정 없이 사실만 수집 (GPU 없어도 항상 성공)."""
    env: dict = {
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "repo_commit": _try(["git", "-C", _REPO, "rev-parse", "--short", "HEAD"]),
    }
    try:
        import numpy  # noqa: F401
        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = None
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda"] = getattr(torch.version, "cuda", None)
        env["cuda_available"] = bool(torch.cuda.is_available())
        if env["cuda_available"]:
            env["device_name"] = torch.cuda.get_device_name(0)
            env["device_capability"] = list(torch.cuda.get_device_capability(0))
            env["device_count"] = torch.cuda.device_count()
    except Exception as e:  # torch 자체가 없는 저작 호스트
        env["torch"] = None
        env["torch_import_error"] = f"{type(e).__name__}: {e}"[:200]
        env["cuda_available"] = False
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None
    try:
        import vllm
        env["vllm"] = vllm.__version__
    except Exception:
        env["vllm"] = None
    smi = _try(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader"])
    env["nvidia_smi"] = smi
    env["compute_sanitizer"] = find_compute_sanitizer()
    env["torch_cuda_arch_list"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
    env["in_venv"] = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    env["python_exe"] = sys.executable
    return env


def check_preconditions(env: dict, cpu_dry_run: bool = False) -> tuple[bool, list[dict]]:
    """(진행가능 여부, 항목별 결과). cpu_dry_run이면 GPU 항목은 전부 SKIP 처리."""
    checks: list[dict] = []

    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    add("python>=3.10", "PASS" if sys.version_info >= (3, 10) else "FAIL", env["python"])

    if cpu_dry_run:
        for name in ("torch+cuda", "gpu_visible", "sm80", "triton", "compute-sanitizer"):
            add(name, "SKIP", "--cpu-dry-run: GPU 사전조건 검사 생략 (참조 경로만 실행)")
        return True, checks

    # venv 미사용은 torch/vllm 미탐의 1순위 원인 — 가장 먼저 보여준다
    if env.get("in_venv"):
        add("venv", "PASS", env.get("python_exe", ""))
    else:
        add("venv", "SKIP",
            f"시스템 python({env.get('python_exe')}) 실행 중 — GPU가 있어도 torch/vllm 이 "
            "여기 없으면 연쇄 FAIL. 해결: `bash a100-kit/bootstrap.sh --stage2` (자동 구성) "
            "또는 RUNBOOK §2/§3 venv activate 후 재실행")

    if not env.get("torch"):
        add("torch+cuda", "FAIL",
            "torch import 실패 — GPU 유무는 torch 를 통해 보므로 이 항목이 죽으면 GPU 검사도 "
            "연쇄 FAIL. 해결: `bash a100-kit/bootstrap.sh --stage1`(wheel, 빠름) 또는 "
            "`--stage2`(kimi-k3 소스빌드) / 이미 구성했다면 해당 venv activate 후 재실행")
    elif not env.get("cuda_available"):
        add("torch+cuda", "FAIL",
            f"torch {env['torch']} 는 있으나 CUDA 사용 불가 — nvidia-smi 동작 여부/드라이버 버전"
            "(RUNBOOK §1.1·§6.1)/CUDA_VISIBLE_DEVICES 확인. pip 로 해결 안 되는 항목")
    else:
        add("torch+cuda", "PASS", f"torch {env['torch']} / cuda {env.get('torch_cuda')}")

    if env.get("cuda_available"):
        add("gpu_visible", "PASS", f"{env.get('device_count')}× {env.get('device_name')}")
        cap = tuple(env.get("device_capability") or ())
        if cap == (8, 0):
            add("sm80", "PASS", f"compute capability {cap} (SM80)")
        else:
            add("sm80", "FAIL",
                f"capability {cap} ≠ (8,0) — 이 킷의 판정(tol/PERF 목표)은 SM80(A100) 전용, 타 아치 결과는 무효")
    else:
        add("gpu_visible", "FAIL", "CUDA 디바이스 없음")
        add("sm80", "FAIL", "GPU 없음 — SM80 확인 불가")

    add("triton", "PASS" if env.get("triton") else "FAIL",
        env.get("triton") or "triton 미설치 (k01~k04 Triton 커널 실행 불가) — torch 설치에 "
        "수반됨. 해결: bootstrap 또는 해당 venv 에서 `pip install triton`")

    # vllm 은 env_check 를 막지 않는다(k06/k07 은 불필요) — 단 k01~k05 는 개별 rc=2 로 skip 됨
    add("vllm", "PASS" if env.get("vllm") else "SKIP",
        env.get("vllm") or "vllm import 불가 — k01~k05 하네스가 각자 rc=2 로 skip 된다. "
        "해결: `bash a100-kit/bootstrap.sh --stage2` (kimi-k3 소스빌드 venv)")

    # sanitizer는 없어도 진행은 가능(SANITIZER 게이트가 SKIP 사유를 남김) → WARN 성격의 SKIP
    if env.get("compute_sanitizer"):
        add("compute-sanitizer", "PASS", env["compute_sanitizer"])
    else:
        add("compute-sanitizer", "SKIP",
            "미발견 — SANITIZER 게이트는 SKIP(사유 기록)으로 진행됨. CUDA 툴킷 bin/ 확인 (RUNBOOK §1.5)")

    ok = all(c["status"] != "FAIL" for c in checks)
    return ok, checks


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    dry = "--cpu-dry-run" in argv
    quiet = "--quiet" in argv
    env = collect_env()
    ok, checks = check_preconditions(env, cpu_dry_run=dry)
    result = {"mode": "cpu-dry-run" if dry else "gpu", "ok": ok, "checks": checks, "env": env}
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    elif not quiet:
        print(f"[env_check] mode={result['mode']}")
        for c in checks:
            print(f"  {c['status']:4s} {c['check']:20s} {c['detail']}")
    if ok:
        if not quiet and not as_json:
            print("[env_check] 사전조건 충족 — 진행")
        return EXIT_OK
    if not as_json:
        print("[env_check] GPU 사전조건 미충족 — 커널 실행 skip. "
              "GPU 없는 호스트에서는 --cpu-dry-run 으로 참조 경로만 검증하라.", file=sys.stderr)
    return EXIT_PRECOND


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
