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

## 2. openai 티어 `k31-sm80-openai:a3c14efb-sm80-v1` — **PASS (CPU 모드 17/17, 경고 0)** (2026-11-10 01:29, 약 10분)

| 항목 | 값 |
|---|---|
| 이미지 ID | `sha256:f5ea76f4f007019ee2098d886582bf79faaa1c506d885292e8292f432bd2e40b` |
| 크기 | 20,726,581,453 B (20.7 GB) |
| base | `k31-sm80-kernelkit:full-a3c14efb` (`io.k3.base_digest` 라벨로 고정, §1) |
| patchset | `mxfp4-marlin-moe-sm80-main+triton-mla-sparse-sm80+k31-sm80-dsa-glue` (`/opt/k31-patches/`, `patches.sha256`) |
| 빌드 로그 | `docker/logs/k31-openai-20261110-011957.log`(gitignored) |

### `verify_k31_image.py`(CPU 모드) 결과 원문 요지

| 섹션 | 검사 | 결과 |
|---|---|---|
| static | torch-cuda(torch 2.14.0+cu130, cuda 13.0, arch 목록에 sm_80) · overlay-manifest(10파일 일치) · shipped-patches(3개, `patches.sha256` 일치) · moe-so(설치 `.so` == marlin 패치 빌드본) · deepgemm-absent · kimi-k3-dsa-code(`dsa.py` 존재) · compute-sanitizer | 전부 PASS |
| import(CPU) | `vllm.entrypoints.openai.api_server` · `kimi_k3_tool_parser` · `kimi_k3_reasoning_parser` · `vllm.models.kimi_k3.nvidia.dsa` · `vllm.v1.attention.ops.mqa_logits_triton` · `…triton_mla_sparse_kernel` · `torch.ops._moe_C.moe_wna16_marlin_gemm` 등록 · `AttentionBackendEnum.TRITON_MLA_SPARSE` → `TritonMLASparseBackend`(mla+sparse, sm80) | 전부 PASS |
| cli | `vllm serve --help` 필수 플래그 25개 | PASS |
| gpu | — | **SKIP**(CPU 모드: GPU 동작 미검증 — A100 노드에서 `--gpu`로 재실행) |

`K31-OPENAI-VERIFY-PASS image=k31-sm80-openai:a3c14efb-sm80-v1 id=sha256:f5ea76f4… mode=cpu checks=17 warnings=0`

### 정직한 상한
이 PASS는 "이미지가 계약대로 조립됐고 CPU에서 import된다"까지다. 커널 실행·백엔드 선택·서빙은 사내 A100에서 `python3 docker/verify_k31_image.py <image> --gpu` → `run/21` → lane 11 smoke 순으로 판정한다.

## 3. Release — **UPLOAD-PASS** (2026-11-10 01:43)

| 항목 | 값 |
|---|---|
| URL | https://github.com/jimmylegendary/k3-sm80/releases/tag/openai-k31-a3c14efb-sm80-v1 (private repo) |
| 자산 | `k31-sm80-openai-a3c14efb-sm80-v1.tar.zst.part-aa/ab/ac` 각 1,992,294,400 B · `part-ad` 197,354,693 B (합 ≈ 6.17 GB, `docker save` 20.7 GB를 zstd -10 압축) · `SHA256SUMS`(460 B) · `image-inspect.json`(7,910 B) |
| 로컬 사본 | `docker/releases/openai-k31-a3c14efb-sm80-v1/`(gitignored) |
| 사내 반입 | `bash run/11-load-k31-image.sh`(gh 자동 다운로드 → `sha256sum -c` → 결합 → `docker load`, 멱등·part 오염 검사). 수동: `cat …part-* | zstd -d | docker load` — [PULL-K31.md](/home/jimmy/repos/k3-sm80/docker/PULL-K31.md) |
| 이미지 ID 대조 | load 후 `docker image inspect k31-sm80-openai:a3c14efb-sm80-v1 --format '{{.Id}}'` = `sha256:f5ea76f4f007019ee2098d886582bf79faaa1c506d885292e8292f432bd2e40b` 이어야 한다(§2). `verify_k31_image.py --gpu`가 같은 값을 검사한다 |

K3 Release(`openai-0498dc7e-sm80-v3`)와 `run/10`은 변경 없음.
