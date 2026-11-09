# K3.1 이미지 빌드 게이트 판정 — full 티어 (개발 호스트, GPU 없음)

K3용 [GATE-RESULT.md](/home/jimmy/repos/k3-sm80/docker/GATE-RESULT.md)와 별개 문서다. 이 게이트의 목적은 GPU 없는 호스트에서
"핀 `a3c14efb`가 SM80 단일 아치로 소스빌드되고, 기대한 커널만 빌드되고(DeepGEMM·FlashKDA 제외), 런타임 계약(버전·확장·sanitizer)을 만족한다"를
컨테이너 안에서 실증하는 것이다. GPU 실행 판정은 사내 A100 라운드(`docker/verify_k31_image.py --gpu`, `run/21`)에서만 한다.

## 1. full 티어 `k31-sm80-kernelkit:full-a3c14efb` — **PASS** (2026-11-10 01:18, 7,052 s, MAX_JOBS=6)

| 항목 | 값 |
|---|---|
| 이미지 ID (manifest list) | `sha256:89e1aa1e8ce0004076fcc04a0ad2b1ba830521afb2942f9a808f9e121f300324` |
| 크기 | 20,655,584,858 B (20.7 GB) |
| 베이스 | `nvidia/cuda:13.0.3-devel-ubuntu24.04` (build) / `nvidia/cuda:13.0.3-base-ubuntu24.04` (runtime) — 핀 CMake가 g++ ≥ 13 요구 → 24.04 |
| vLLM | `vllm 0.33.0rc1.dev1090+ga3c14efbc` @ `a3c14efbc77c018e8bf579935095a87721349e05`, `pip install -v . --no-build-isolation`, `TORCH_CUDA_ARCH_LIST=8.0` |
| 스택 | torch `2.14.0+cu130` · torchvision `0.29.0+cu130` · triton `3.8.0` · nvcc 13.0 · python 3.12 · 컴파일 확장 `.so` 7개 |
| 라벨 | `io.k3.cuda_arch=sm80`, `org.opencontainers.image.revision=vllm@a3c14efb…`, `version=24.04` |
| 빌드 로그 | `docker/logs/k31-full-20261109-232105.log`(gitignored, 20,001줄) · 이미지 내 `/opt/build-k31.log.gz` |

### 로그 게이트(빌드 중 RUN 단계로 강제 — 실패 시 빌드 중단)

| 게이트 | 로그 원문 |
|---|---|
| pre-build | `pre-build gate ok: torch 2.14.0+cu130 \| cuda 13.0 \| torchvision 0.29.0+cu130 \| nvcc 13.0` |
| 단일 아치 | `-- CUDA target architectures: 8.0` (configure 2회 모두) |
| DeepGEMM 미빌드(의도) | `-- DeepGEMM will not compile: unsupported CUDA architecture 8.0` → SM80 indexer 경로는 패치의 Triton fallback으로 간다 |
| FlashKDA 미빌드(의도) | `-- FlashKDA will not compile: CUDA >=12.0 and a supported architecture (SM90, SM10x, or SM12x) are required` → KDA는 Triton 폴백 |
| SM90+ 전용 커널 제외 | Machete · dsv3_fused_a_gemm · scaled_mm_c3x(sm90/100/120) · grouped_mm_c3x · moe_data · W4A8 전부 "Not building … no compatible archs" |
| post-install | `gate ok: vllm 0.33.0rc1.dev1090+ga3c14efbc \| torch 2.14.0+cu130 \| triton 3.8.0 \| .so 7 \| commit a3c14efb` |
| sanitizer | runtime 단계 `compute-sanitizer --version` 성공 |

### 주의
- 이미지의 Triton은 **3.8.0**이다. 호스트 CPU 인터프리터 대조는 venv의 3.9.0에서 했고(`kernel-kit/common/triton_interp.py` 보정은 3.9 전용), GPU 컴파일 경로는 영향이 없을 것으로 보지만 **A100에서 k11/k12 커널의 3.8.0 컴파일은 미확인**이다. W1의 sm_80 PTX 컴파일 확인도 호스트 3.9.0 기준이다.
- 이 티어는 패치 전 base다. 사내 반입 대상은 아래 openai 티어다.

## 2. openai 티어 `k31-sm80-openai:a3c14efb-sm80-v1` — (빌드 진행 중, 2026-11-10 01:19 시작)

패치 3개(`docker/patches-k31/`) 적용 → `_moe_C_stable_libtorch` 재컴파일 → Python overlay 10파일 → `verify_k31_image.py`(CPU 모드: static·import·cli, gpu 섹션은 SKIP 명시).
결과는 완료 후 이 절에 기입한다.
