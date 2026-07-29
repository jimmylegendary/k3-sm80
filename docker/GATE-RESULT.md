# 컨테이너 gate 판정 — triton 티어 (PLAN §S2 "반입 전 필수")

- 판정: **PASS** (2026-07-29, 개발 호스트 · GPU 없음 — 이 gate 의 목적 자체가 GPU 없는 환경에서
  이미지·vendor 의 무결성을 컨테이너 안에서 실증하는 것)
- 수정 내역: **없음** (재시도 0회 — 빌드·dry-run·import 실증 전부 1회차 green)

## 1. 이미지

| 항목 | 값 |
|---|---|
| 태그 | `ghcr.io/jimmylegendary/k3-sm80-kernelkit:triton` = `:triton-20260729` |
| digest (manifest list = Image ID) | `sha256:560f65bc576e40f42c36bd772cad30fb6916ac0571c1af7f3acb9da46ff05870` |
| 베이스 | `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime@sha256:6acf597eeb8e376a96580dde4952f37cc017fef732bb40bfc73f28f25e3f64b4` |
| 크기 | content 3.9GB (disk usage 11.3GB) |
| 빌드 | `bash docker/build.sh --triton-only` — 로그 `docker/logs/build-triton-20260729-141448.log` |
| 스택 | torch `2.13.0+cu126` · triton `3.7.1` · numpy `2.5.1` · python `3.12.3` |

주: digest 는 로컬 빌드 산출(containerd 스토어의 manifest list). ghcr push 전이므로
레지스트리 digest 는 push 후 `docker/push.sh` 가 확정 — 사내 반입 시 대조 기준은 push 산출물.

빌드 게이트(이미지 내장, Dockerfile.triton)도 이 빌드에서 통과: torch==2.13.0/cu12.x/triton/numpy
assert + **vllm 존재 시 빌드 실패하는 역게이트** (`import vllm` 성공이 곧 FAIL).

## 2. 컨테이너 gate: `run_all.sh --cpu-dry-run` — exit 0

실행 (build.sh 내장 gate 1회 + 명시 재실행 1회, 동일 결과):

```
docker run --rm -v /home/jimmy/repos/k3-sm80:/work -u "$(id -u):$(id -g)" -e HOME=/tmp \
  ghcr.io/jimmylegendary/k3-sm80-kernelkit:triton bash /work/kernel-kit/run_all.sh --cpu-dry-run
# → EXIT_CODE=0
```

집계 (`kernel-kit/kernel-report/SUMMARY.md`, 컨테이너 안에서 재생성):

- 종합: **PASS 0 / FAIL 0 / INCOMPLETE 7 / MISSING 0** — 7/7 커널 전부
  `INCOMPLETE`(NUMERIC/SANITIZER/SHAPES/PERF 전 게이트 SKIP)
- dry-run 상한 = INCOMPLETE 준수, **PASS 사칭 없음** (k04 의 `selfcheck=PASS` 는
  참조구현 scaled 자체점검 로그일 뿐 게이트 판정 아님 — overall 은 INCOMPLETE)
- 마지막 줄: `[run_all] 종합: FAIL 없음 (dry-run 상한=INCOMPLETE)`
- 실행 증거: `kernel-report/env.json` 의 `host=e794a93e3355`(컨테이너 hostname),
  `torch=2.13.0+cu126`, `python_exe=/usr/bin/python3`, `repo_commit=beb7859`
- 로그: `docker/logs/gate-triton-20260729-141452.log` (build.sh 내장 실행분)

## 3. vendor import 실증 — vLLM 없이 torch+triton 만으로 로드

같은 이미지에서 (전체 출력 원문):

```
$ docker run --rm -v /home/jimmy/repos/k3-sm80:/work -u "$(id -u):$(id -g)" -e HOME=/tmp \
    ghcr.io/jimmylegendary/k3-sm80-kernelkit:triton python3 -c "<13심볼 전수 import + vllm 부재 assert>"
vllm find_spec: None (image에 vllm 부재 확인)
vendor import: 13/13 심볼 callable (VENDOR_PIN=0498dc7ea9204f1863b46515816f5d8d458e01e7)
torch=2.13.0+cu126 cuda=12.6 triton=3.7.1
sys.modules 에 vllm* 0건 — vLLM 없이 torch+triton 만으로 로드 성공
# rc=0
```

- import 한 13심볼 = `vendor/__init__.py` `_EXPORTS` 전량: `chunk_kda`, `chunk_kda_fwd`,
  `chunk_kda_with_fused_gate`, `chunk_kda_with_fused_gate_fwd`, `fused_kda_gate`,
  `fused_kda_gate_chunk_cumsum`, `fused_recurrent_kda`, `fused_recurrent_kda_fwd`,
  `fused_recurrent_kda_packed_decode` (k01·k02), `causal_conv1d_fn`, `causal_conv1d_update`
  (k03), `decode_attention_fwd` (k04), `situ_and_mul_native` (k07)
- PEP 562 lazy 를 실체화한 것이므로 fla_kda/fla_common/mamba_ops/attn_ops/situ +
  `_vllm_shims` 전 체인이 실제 모듈 로드됨 (triton `@triton.jit` 데코레이션 포함)
- `importlib.util.find_spec('vllm') is None` + import 후 `sys.modules` 에 `vllm*` 0건
  → "triton 이미지에 vllm 이 없음" 과 "vendor 가 vllm 을 요구하지 않음" 동시 증명

`vendor/check_no_vllm.py` 컨테이너 내 재실행 (rc=0):

```
[check_no_vllm] green — 24 파일: py_compile + vllm import 0건 + 외부 의존 ⊆ ['numpy', 'packaging', 'torch', 'triton']
```

## 4. 판정 요약

PLAN §S2 컨테이너 gate 조건 대조:

| 조건 | 결과 |
|---|---|
| triton 이미지 안 `run_all.sh --cpu-dry-run` green | exit 0, FAIL 0 (7/7 INCOMPLETE 상한) |
| vllm import 0건 (k05 제외) | check_no_vllm green 24파일 + find_spec None + sys.modules 0건 + Dockerfile 역게이트 |

→ **triton 티어 반입 가능.** 남은 것: `docker/push.sh` 로 ghcr push(레지스트리 digest 확정)
및 full 티어(`Dockerfile.full`, k05/S4 용)의 별도 빌드·gate.
