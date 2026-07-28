# A100 반입 프로브 킷 — RUNBOOK (vLLM SM80 소스 빌드 + 프로브 실행)

작성: 2026-07-28 · k3-sm80 Phase 1 · 러너: Jimmy (사내 A100 노드)
근거: `recon/vllm-upstream.md`, `recon/kda-fla.md`, `recon/k3-config.md` (전부 2026-07-28 스냅샷)

## 0. 버전 고정표 (이 킷의 전제 — 변경 시 프로브 판정 무효)

| 대상 | 고정값 | 근거/출처 |
|---|---|---|
| vLLM 릴리스 wheel (Stage 1) | **`vllm==0.26.0`** (2026-07-27 릴리스; torch 2.11.0, flashinfer-python 0.6.14 자동 수반) | https://pypi.org/pypi/vllm/0.26.0/json (2026-07-28 확인: cp38-abi3 manylinux x86_64 wheel 존재) |
| vLLM kimi-k3 브랜치 (Stage 2) | **`0498dc7ea9204f1863b46515816f5d8d458e01e7`** (PR #50000 head, 2026-07-28T04:47Z) | https://github.com/vllm-project/vllm/pull/50000 · recon/vllm-upstream.md |
| vLLM main (참고 기준) | `d223c900d85224c02f2162ee2c757a769e99f519` | recon/vllm-upstream.md |
| kimi-k3 브랜치 의존성 | torch==**2.13.0**, flashinfer-python==**0.6.15.post1**(+`--extra-index-url https://flashinfer.ai/whl/`), **flash-linear-attention==0.5.0**(pip, "# KDA" 명시) | `requirements/cuda.txt` @ 0498dc7 (2026-07-28 fetch) |
| CUDA 툴킷 (Stage 2 빌드) | **13.0.x** (공식 Dockerfile `ARG CUDA_VERSION=13.0.3`), Python **3.12** (Dockerfile), requires-python `>=3.10,<3.15` | `docker/Dockerfile`, `pyproject.toml` @ 0498dc7 |
| probe_35922 모델 | **`RedHatAI/Qwen3-30B-A3B-quantized.w4a16`** @ `1c842a5000febac4b335c94a04789a9d4756a18e` (공개, ~16.7GB) | HF API 2026-07-28 확인 (issue #35922는 로컬 양자화 ckpt 사용 — 이건 동일 포맷의 공개 대체재) |
| probe_35922 FP4 arm 모델 | **`openai/gpt-oss-20b`** @ `6cee5e81ee83917806bbde320786a8fb61efebee` (공개, MXFP4 MoE) | HF API 2026-07-28 확인 |
| probe_k3_modules 모델 | `inference-optimization/Kimi-K3-0.18B`는 **존재하지 않음**(HF API 401, 2026-07-28) → 대안 = `moonshotai/Kimi-K3` @ **`9f62e4e9fffbd0a83ddd60e1c209d828994b3569`**(공개, non-gated)의 config/tokenizer/remote-code만 받아 **축소 config + `--load-format dummy`** | recon/k3-config.md · probe_k3_modules.py 내장 |
| FlashKDA | SM80 **빌드 자동 스킵**(cmake 경고만; `SUPPORTED_CUDA_ARCHS=["90a","100a","103a","120a"]`) — 조치 불필요 | recon/vllm-upstream.md (2), recon/kda-fla.md §2 |
| FlashInfer pre-release (#4178/#4180) | **SM80 경로에 불필요.** B300/B200 도커 이미지 전용 의존성. 설치 금지(표준 0.6.15.post1만) | recon/vllm-upstream.md (4) |

검증 상태(2026-07-28, 호스트에서 raw fetch로 재확인 완료): ① 프로브가 호출하는 커널 시그니처 전부 pin 커밋 원본과 대조 일치 — main `chunk_kda`/`fused_recurrent_kda`(kda.py L1458/L109), k3 `chunk_kda_with_fused_gate`(raw_g/raw_beta/A_log/g_bias/lower_bound, chunk.py L776)/`fused_recurrent_kda`(raw 입력+fuse_gate, in-kernel l2norm 상시, fused_recurrent.py L429)/게이트 수식(`lower_bound*sigmoid(exp(A_log)*(g+bias))`, beta=sigmoid) ② S1 대상 3함수 시그니처(kda.py L58/L85/L148) ③ `EngineArgs.kernel_config`(dict→KernelConfig, arg_utils.py L755)와 `moe_backend∈{triton,marlin,emulation,…}`, `attention_backend` 문자열→enum 변환(arg_utils.py L2313) ④ requirements/Dockerfile pin(§0 표) ⑤ v0.26.0 태그에 vendored KDA·marlin_moe·kernel.py 존재 ⑥ HF 3개 체크포인트 revision 실재(전부 non-gated) ⑦ `VLLM_ENABLE_V1_MULTIPROCESSING` env 유효(arm B 인프로세스 관측 전제).

주의: kimi-k3 브랜치는 머지 전까지 주 단위로 움직인다(force-push 가능). **반드시 위 커밋으로 checkout**하고,
checkout 실패(force-push로 커밋 소실) 시 `git fetch origin pull/50000/head`로 새 head를 받되
**새 커밋 해시를 결과 로그에 기록**하고 DECISION-TREE의 "브랜치 드리프트" 절차를 따른다.

## 1. 사전조건 점검 (노드에서 가장 먼저 실행)

```bash
# ---- 1.1 GPU / 드라이버 ----
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
# 기대: A100 ... , 8.0 , 81920 MiB, driver >= 580.xx
#  - torch 2.11/2.13의 PyPI 기본 wheel은 CUDA 12.8 또는 13.0 빌드다. 어느 쪽인지는
#    설치 후 `python -c "import torch; print(torch.version.cuda)"`로 확인하고,
#    CUDA 13.x면 driver r580+, CUDA 12.8이면 r570+ 필요(미달 시 아래 6.1 참조).

# ---- 1.2 컴파일러 (Stage 2 소스빌드에만 필요) ----
nvcc --version          # 기대: release 13.0 (Dockerfile ARG CUDA_VERSION=13.0.3와 major.minor 일치)
gcc --version           # gcc 11~13 권장
cmake --version         # >= 3.26.1 (pyproject [build-system]); 없으면 pip로 설치됨
python3 --version       # 3.10~3.14 허용, 3.12 권장(공식 Dockerfile 기준)

# ---- 1.3 네트워크 (docker registry 외 전부 열려있다는 전제의 실검증) ----
curl -sSI https://pypi.org/simple/ | head -1                  # PyPI
curl -sSI https://huggingface.co | head -1                    # HF
curl -sSI https://github.com | head -1                        # GitHub
curl -sSI https://flashinfer.ai/whl/ | head -1                # flashinfer wheel index (Stage 2)
#  flashinfer.ai가 막혀 있으면: flashinfer-python 0.6.15.post1은 PyPI에도 있음(2026-07-28 확인)
#  → --extra-index-url 없이도 설치됨. flashinfer-cubin은 PyPI에 없으나 setup.py가
#    install_requires에서 제외하므로(주석 명시) `pip install -e .`에는 영향 없음.

# ---- 1.4 디스크 ----
df -h "$HOME" /tmp
# 필요량: HF 캐시 ~35GB(W4A16 16.7GB + gpt-oss-20b ~14GB + K3 소형파일) + 소스빌드 ~30GB + venv 2벌 ~20GB

# ---- 1.5 compute-sanitizer (probe_35922 arm B) ----
which compute-sanitizer || ls /usr/local/cuda*/bin/compute-sanitizer
# CUDA 툴킷 동봉. 없으면 arm B는 자동 스킵된다(스크립트가 처리).
```

## 2. Stage 1 — 릴리스 wheel 경로 (빌드 없이 당일 결과)

목적: **probe_35922 전체 + probe_kda_fla(main vendored 사본)**. Marlin MoE crash와 KDA Triton 수치는
릴리스 wheel(0.26.0)로 검증 가능하다 — 소스빌드를 기다릴 필요가 없다.
(0.26.0 wheel의 컴파일 아치에 sm80 포함: `MARLIN_MOE_ARCHS="8.0+PTX;12.0f"`가 릴리스 빌드에 그대로 적용 — recon/vllm-upstream.md (5))

```bash
mkdir -p ~/k3-sm80-work && cd ~/k3-sm80-work
python3 -m venv venv-probe && source venv-probe/bin/activate
pip install -U pip
pip install vllm==0.26.0        # torch 2.11.0 등 자동 수반. 수 GB 다운로드.

# ---- 설치 검증 ----
python - <<'EOF'
import torch, vllm
print("vllm:", vllm.__version__)
print("torch:", torch.__version__, "| torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("capability:", torch.cuda.get_device_capability())     # 기대 (8, 0)
from vllm.third_party.flash_linear_attention.ops.kda import chunk_kda  # vendored KDA 존재 확인
print("vendored KDA: OK")
# Marlin MoE 파이썬 래퍼 위치는 v0.26.0/main 기준 experts/marlin_moe.py (2026-07-28 raw 확인)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
print("marlin moe import: OK")
EOF

# ---- 스모크: 아무 소형 모델로 e2e 1회 ----
python - <<'EOF'
from vllm import LLM, SamplingParams
llm = LLM(model="facebook/opt-125m", enforce_eager=True, gpu_memory_utilization=0.3)
print(llm.generate(["Hello"], SamplingParams(max_tokens=8))[0].outputs[0].text)
EOF
```

Stage 1 프로브 실행:

```bash
cd <이 킷 디렉터리>   # k3-sm80/a100-kit
bash probe_35922.sh                       # arm A~E 전부 (FP4 arm 포함; 스킵은 PROBE_ARMS로)
python probe_kda_fla.py --impl main       # main vendored 사본 수치검증
python probe_kda_fla.py --impl main --perf   # (선택) 성능 스모크
```

## 3. Stage 2 — kimi-k3 브랜치 소스 빌드 (K3 모델 코드 경로)

목적: **probe_k3_modules + probe_kda_fla(k3 사본)** + 이후 Phase 2 본작업.
K3 모델 코드(`vllm/models/kimi_k3/`)는 main에 없고 이 브랜치에만 있다(recon (4)). **별도 venv 필수**
(Stage 1과 torch 버전이 다르다: 2.11.0 vs 2.13.0).

```bash
cd ~/k3-sm80-work
python3 -m venv venv-k3 && source venv-k3/bin/activate
pip install -U pip

# ---- 소스 확보 + 커밋 고정 ----
git clone https://github.com/vllm-project/vllm.git vllm-k3
cd vllm-k3
git fetch origin pull/50000/head:kimi-k3-pr
git checkout 0498dc7ea9204f1863b46515816f5d8d458e01e7 \
  || { echo "PIN 커밋 소실(force-push 의심) — 새 head로 진행하되 해시 기록"; \
       git checkout kimi-k3-pr; git rev-parse HEAD | tee ../K3_BRANCH_ACTUAL_COMMIT.txt; }
git rev-parse HEAD    # 결과 디렉터리에 반드시 기록

# ---- 빌드 ----
# TORCH_CUDA_ARCH_LIST=8.0 : A100 전용. Marlin MoE는 CMake가 "8.0+PTX;12.0f"와 교집합을 취해
#   sm80 커널을 빌드한다(INT4 + bf16-act NVFP4/MXFP4 전부 sm80_kernel_*.cu에 인스턴스화 — recon (5)).
# FlashKDA: cmake가 "FlashKDA will not compile: ... SM90, SM10x, or SM12x ..." 경고 후 스킵 — 정상.
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=32                    # 노드 코어 수에 맞춰 조절 (RAM 부족 시 16으로)
export CCACHE_DIR=$HOME/.ccache       # ccache 있으면 재빌드 가속 (없어도 무방)
# 빌드 의존성 (경로 주의: 이 커밋에는 requirements/build.txt가 없고 build/cuda.txt다 —
# torch==2.13.0 포함. --no-build-isolation은 빌드 시점에 venv의 torch를 요구하므로 선설치 필수)
pip install -r requirements/build/cuda.txt
pip install -r requirements/cuda.txt   # 런타임 pin 일괄(FLA 0.5.0·flashinfer 0.6.15.post1 포함)
pip install -e . --no-build-isolation 2>&1 | tee ../build-k3.log
# 소요: 64코어 기준 대략 40~90분. 실패 시 build-k3.log 끝부분과 §6 트러블슈팅 참조.

# ---- 빌드 검증 ----
python - <<'EOF'
import torch, vllm
print("vllm:", vllm.__version__, "| torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("capability:", torch.cuda.get_device_capability())          # (8, 0)
import vllm.models.kimi_k3                                        # K3 모델 코드 = 브랜치 증명
print("kimi_k3 package: OK")
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate
print("K3 KDA triton ops: OK")
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print("KimiK3ForConditionalGeneration registered:", "KimiK3ForConditionalGeneration" in archs)
# fused_kda_decode(SM90+ CUDA 커널)는 SM80 빌드에 없어야 정상:
print("fused_kda_decode compiled:", hasattr(torch.ops._C, "fused_kda_decode"), "(False가 정상)")
EOF

# ---- 업스트림 KDA 테스트 (A100 수치검증 — FLA CI는 H100 전용이라 이게 첫 SM80 공식 대조) ----
pip install pytest
pytest tests/kernels/test_kda.py -x -q                  # main 계열 vendored 사본 대상
pytest tests/models/kimi_k3/test_kda.py -x -q           # K3 전용 757줄 테스트 (경로는 브랜치 기준)
pytest tests/models/kimi_k3/test_kda_metadata.py -x -q
```

Stage 2 프로브 실행:

```bash
cd <이 킷 디렉터리>
python probe_kda_fla.py --impl k3          # K3 사본 수치검증 (T4/T5/T6)
python probe_kda_fla.py --impl fla         # (선택) pip fla 0.5.0 교차대조
python probe_k3_modules.py                 # K3 모듈 dummy-weight e2e
PROBE_ATTN_BACKEND=TRITON_MLA python probe_k3_modules.py --stages S3   # (선택) MLA 백엔드 강제 시도
```

## 4. 실행 순서 요약 + 결과 수집

```
1. §1 사전조건 → 2. Stage 1 설치 → 3. probe_35922.sh → 4. probe_kda_fla --impl main
→ 5. Stage 2 빌드 → 6. pytest KDA 3종 → 7. probe_kda_fla --impl k3 → 8. probe_k3_modules.py
```

각 프로브는 `results/` 아래 타임스탬프 디렉터리에 로그와 `SUMMARY.txt`(기계판독 `PROBE...|` 라인)를 남긴다.
반출:

```bash
tar czf k3-sm80-probe-results-$(date +%Y%m%d).tgz results/ ~/k3-sm80-work/build-k3.log \
    ~/k3-sm80-work/K3_BRANCH_ACTUAL_COMMIT.txt 2>/dev/null
```

판정과 다음 행동은 **DECISION-TREE.md**를 따른다.

## 5. FLA / FlashInfer 처리 요약 (스펙 Phase 1 질문에 대한 답)

- **FLA 설치**: 별도 조치 불필요. ① vLLM은 KDA Triton 커널을 **소스 벤더링**(main: `vllm/third_party/flash_linear_attention/`, K3 브랜치: `vllm/models/kimi_k3/nvidia/ops/third_party/kda/` — 후자가 전자의 인프라를 import), ② kimi-k3 브랜치 `requirements/cuda.txt`가 `flash-linear-attention==0.5.0`을 pip 의존으로 추가 → **빌드 시 자동 설치**. 교차대조용 외 수동 개입 없음.
- **FlashInfer**: SM80 경로에서 **pre-release 불필요**. 표준 `flashinfer-python==0.6.15.post1`이 requirements로 자동 설치된다. PR #4178/#4180(pre-release)은 B300/B200 도커 전용(MLA/SITU trtllmgen) — recon (4). 설치 시도 금지.
- **FlashKDA**: SM80 빌드 자동 스킵(경고 = 정상). KDA는 prefill/decode 모두 Triton 폴백이 공식 경로 — `additional_config={"kda_prefill_backend":"triton"}`로 명시 고정 권장(recon (2)).

## 6. 트러블슈팅

| 증상 | 원인/조치 |
|---|---|
| 6.1 `CUDA driver version is insufficient` / torch가 GPU 미인식 | 드라이버 < torch CUDA 요구. (a) 드라이버 업그레이드(r580+, 사내 절차), (b) 임시로 `pip install nvidia-cuda-compat-cu13` 후 `LD_LIBRARY_PATH`에 추가(포워드 호환, 데이터센터 GPU 한정), (c) 최후: torch를 cu126 계열로 내리는 건 **vLLM pin 위반이라 금지** — Jimmy 판단으로만. |
| nvcc 13.0 없음 (Stage 2) | (a) 사내 모듈/패키지로 CUDA 13.0 툴킷 설치, (b) 비루트: NVIDIA runfile `--silent --toolkit --toolkitpath=$HOME/cuda-13.0` 후 `PATH`/`CUDA_HOME` 지정, (c) conda 가능하면 `conda install -c nvidia cuda-toolkit=13.0`. nvcc 와 torch.version.cuda의 **major 불일치 상태로 빌드 금지**. |
| `flashinfer.ai` 접속 불가 | §1.3 참조 — PyPI의 flashinfer-python 0.6.15.post1로 충족. `pip install -r requirements/cuda.txt`는 flashinfer-cubin에서 실패할 수 있으나 `pip install -e .`는 영향 없음. |
| 빌드 중 OOM/컴파일러 킬 | `MAX_JOBS`를 16→8로. nvcc 병렬이 램을 크게 먹는다(코어당 ~2GB 가정). |
| `0498dc7` checkout 실패 | force-push. §3 스크립트가 새 head로 폴백 + 해시 기록. DECISION-TREE "브랜치 드리프트" 절차(변경 diff 확인 후 프로브 재실행) 수행. |
| pytest `tests/models/kimi_k3/` 경로 없음 | 브랜치가 테스트를 이동/개명했을 가능성 → `git log --oneline -- tests/ | head`로 확인, `find tests -name "*kda*"`. |
| wheel 설치인데 probe_k3_modules가 S0에서 실패 | 정상 — K3 모델 코드는 wheel(0.26.0)에 없다. Stage 2 venv에서 실행할 것. |
