# 32×A100 클러스터 통신·오버헤드 팩트 (recon)

- 스냅샷: 2026-07-28 (웹 조사 기준일; 모든 실측치는 아래 각 항목의 NCCL/드라이버 버전 고정 참조)
- 용도: Phase 0 hatir 32×A100 시스템 트윈(`hat-schema` `.hw`) + K3 디코드 해석 모델의 통신 파라미터 근거
- 대상: 4노드 × 8×A100-80GB SXM4 (HGX/DGX A100 급), 노드 간 IB 구성은 사내 미상 → 2안

## 0. 모델 입력 요약 (hatir Phase 0에 넣을 값)

| 파라미터 | 보수 | 낙관 | 근거 절 |
|---|---|---|---|
| `nvlink_bw` (노드 내, GPU당) | NCCL 실효 busbw 200 GB/s | 235 GB/s (실측 225.5 상회 여지) | §1 |
| 하드웨어 상한 | 600 GB/s 양방향 = 300 GB/s/방향 (12×NVLink3) | 동일 (공식 스펙) | §1 |
| `ib_bw` (노드 간, 노드당) | **2×HDR200 = 50 GB/s** (실측 busbw 27 GB/s 사례; 최악 1×HDR=25 GB/s) | **8×HDR200 = 200 GB/s** (DGX 표준, GPU당 NIC 1개) | §2 |
| `allreduce_latency` α (소메시지) — 노드 내 8GPU | 20–23 µs (NCCL 기본, 실측) | 3–8 µs (vLLM custom AR / NCCL≥2.27 symmetric) | §3 |
| `allreduce_latency` α — 노드 간 (2노드 16GPU) | 43–46 µs (실측, VM 환경) | ~30 µs (bare-metal 추정) | §3 |
| `launch_overhead` — 커널당 (CUDA graph 無) | 5–10 µs (launch+sync) / back-to-back ~1–4 µs | — | §4 |
| `launch_overhead` — 커널당 (CUDA graph 有) | ~0.5 µs | — | §4 |
| vLLM 디코드 스텝당 CPU 오버헤드 (eager) | 스텝당 ~50–100 µs+, bs=1에서 전체 시간의 20–40% | CUDA graph로 대부분 제거 | §4 |

파생치(모델용 산출, 실측 아님): TP8 기준 레이어당 all-reduce 2회 →
레이어당 통신 α 비용 = 보수 2×20=**40 µs/layer**, 낙관(custom AR) 2×5=**10 µs/layer**.
93레이어 직렬 시 스텝당 3.7 ms(보수) vs 0.93 ms(낙관) — 스펙의 "100 µs/layer" 우려와 정합.
EP dispatch/combine이 노드 경계를 걸치면 a2a 2회/MoE층 × α(노드 간)≈43 µs → 층당 ~86 µs가
추가되므로 노드-로컬 EP 배치가 강제됨.

---

## 1. HGX/DGX A100 8-GPU 노드: NVLink3 + NVSwitch (공식 수치 확인)

- **GPU당 NVLink 총 대역 600 GB/s (양방향)** — A100 공식 datasheet의 Interconnect 항목:
  "NVIDIA NVLink 600 GB/s" (SXM, HGX A100 서버 보드 경유).
  출처: NVIDIA A100 80GB datasheet (JAN21, PDF 직접 확인):
  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf
- **구성: 12개 NVLink3 링크/GPU, 링크당 signal pair 50 Gbit/s** ("50 Gbit/sec per signal
  pair … 12 links in A100 vs 6 in V100 … 600 GB/sec total") → 링크당 25 GB/s/방향,
  GPU당 **300 GB/s/방향**.
  출처: NVIDIA Ampere Architecture In-Depth (developer blog):
  https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- **HGX A100 8-GPU: 2세대 NVSwitch, GPU-to-GPU 600 GB/s, 노드 집계 2.4 TB/s**
  (= 8×300 GB/s/방향). 출처: NVIDIA HGX A100 datasheet:
  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/HGX/a100-80gb-hgx-a100-datasheet-us-nvidia-1485640-r6-web.pdf
- **토폴로지: NVSwitch 6개, 모든 GPU 쌍이 full 12-link(NV12)로 연결** (each GPU fully
  linked … via six NVSwitches; nvidia-smi topo에서 NV12).
  출처: Microway DGX A100 review: https://www.microway.com/hardware/dgx-a100-review-throughput-and-hardware-summary/
  (배경: NVSwitch Hot Chips 34 발표 https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf )
- **NCCL 실효치(실측)**: 8×A100-SXM4-80GB, NCCL 2.18.1 — all_reduce 대메시지(536MB)
  busbw **225.54 GB/s** (algbw 128.88 GB/s). 즉 이론 300 GB/s/방향 대비 ~75%가 ring
  all-reduce 실효 상한. 출처: nccl-tests issue #149:
  https://github.com/NVIDIA/nccl-tests/issues/149

→ 모델값: `nvlink_bw`(collective 실효) 보수 200 / 낙관 235 GB/s. p2p(PP 경계) 전송은
링크 대역이 아니라 소메시지 α가 지배(§3).

## 2. 노드 간 InfiniBand — 통상 구성 + 사내 미상 2안

**"통상"의 기준점 (레퍼런스 시스템 공식 스펙):**

- **DGX A100 표준 = compute fabric용 single-port ConnectX-6 VPI HDR InfiniBand
  200 Gb/s × 8개 (GPU당 1 NIC, rail-optimized)** + 스토리지/이더넷용 dual-port 1–2개.
  출처: NVIDIA DGX A100 datasheet:
  https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf
  및 Microway review(위 §1 링크): "Eight Mellanox 200Gbps HDR InfiniBand/Ethernet
  Single-Port Adapters".
- 클라우드도 동일 패턴: Azure ND A100 v4 — "Each GPU within the VM is provided with
  its own dedicated, topology-agnostic **200 GB/s(=200 Gb/s) NVIDIA Mellanox HDR
  InfiniBand connection**" (VM당 1.6 Tb/s, GPUDirect RDMA 지원).
  출처: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndasra100v4-series
- NCCL은 8-NIC rail 구성을 전제로 PXN(NVLink 경유 NIC 접근)·all2all 메시지 집약을
  최적화함 (all2all 2배 이상 개선). 출처: NCCL 2.12 PXN blog:
  https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/

**사내 미상 → 2안 (모델은 두 안 모두로 판정):**

| 안 | 구성 | 노드당 집계 | GPU당 환산 | 근거 |
|---|---|---|---|---|
| 낙관 | HDR 200 Gb/s × 8 (DGX 표준) | 1.6 Tb/s = **200 GB/s** | 25 GB/s | DGX/Azure 표준 구성 |
| 보수 | HDR 200 Gb/s × 2 | 400 Gb/s = **50 GB/s** | 6.25 GB/s | 사내 축소 구성 흔함; 아래 #949 실측 사례가 정확히 이 구성 |
| (바닥) | HDR × 1 | **25 GB/s** | 3.1 GB/s | 최악 가정용 |

- 보수안의 실측 근거: 2노드 × 8×A100-SXM4, 노드당 **2×HDR**(ConnectX-6, SR-IOV VM)
  에서 all_reduce 대메시지 busbw **27.2 GB/s** (이론 50 GB/s의 ~56%; NIC 단독은
  ~198 Gb/s 정상). 가상화 환경이라 bare-metal은 이보다 나을 수 있음.
  출처: NVIDIA/nccl issue #949: https://github.com/NVIDIA/nccl/issues/949

## 3. NCCL all-reduce / all-to-all latency-bandwidth 모델 파라미터

α-β 모델 t(m) = α + m/B 기준, α(소메시지 평탄 구간) 실측:

- **노드 내 8×A100 (NVLink, NCCL 2.18.1)**: all_reduce 8B **20.65 µs**, 32B 20.77 µs,
  128B 22.40 µs, 1KB ~23.55 µs — 8B~1KB가 평탄 → α ≈ **20–23 µs**.
  출처: nccl-tests #149 (§1 링크, 표 원문).
- **노드 간 2노드 16GPU (2×HDR/node, VM)**: all_reduce 8B **43.04 µs**, 64B 42.53 µs,
  1KB 46.39 µs, 4KB 55.32 µs → α ≈ **43–46 µs** (수십 µs 근거). VM/SR-IOV 오염분을
  감안해 bare-metal 낙관치 ~30 µs로 둔다(GPUDirect RDMA p2p 자체는 3.4 µs까지 가능
  — NVIDIA Network IO blog: https://developer.nvidia.com/blog/accelerating-io-in-the-modern-data-center-network-io/ ).
  출처: NVIDIA/nccl #949 (§2 링크).
- **프로토콜별 per-hop 지연 (NCCL 내부 상수)**: LL ≈ **1 µs**, LL128 ≈ **2 µs**,
  Simple ≈ **6 µs**/hop; 대역 효율 LL 25–50%, LL128 ~95%, Simple ~peak.
  ring 8-GPU all-reduce는 2(n−1)=14 hop → LL로도 ~14 µs+가 하한이라 위 20 µs 실측과
  정합. 출처: Demystifying NCCL, arXiv:2507.04786 (Table I):
  https://arxiv.org/abs/2507.04786
- **소메시지 지연 절감 경로 (낙관치 근거)**:
  - vLLM custom all-reduce(one-shot/two-shot) 계열: 노드 내 소메시지에서 NCCL 대비
    수 µs대로 단축; MSCCL++는 NCCL 대비 소메시지 최대 3.8× 지연 절감, A100 e2e
    디코드 평균 1.11× 개선. 출처: MSCCL++ arXiv:2504.09014
    https://arxiv.org/pdf/2504.09014 ; one-shot 계열 µs 비교(세대는 GB200):
    arXiv:2607.16100 https://arxiv.org/html/2607.16100v1
  - NCCL 2.27 symmetric-memory low-latency 커널: 소메시지 지연 최대 **9×** 절감
    (NVL8=DGX/HGX 8-GPU 도메인에도 적용, 중소 메시지 2.5×). 출처:
    https://developer.nvidia.com/blog/enabling-fast-inference-and-resilient-training-with-nccl-2-27/
  → 낙관 α(노드 내) = 20 µs/9 ≈ 2 µs~수 µs → 모델에는 **3–8 µs** 밴드 사용.
- **디코드 통신 체급 참고**: Llama-70B급 TP 디코드 all-reduce 메시지는 ~256 KiB
  규모로 α 지배 구간; 다노드 32GPU에서 통신이 스텝 시간의 >50%까지 비대해지는
  실측 보고. 출처: UMD PSSG "Beyond NCCL" (Perlmutter A100):
  https://pssg.cs.umd.edu/blog/2025/beyond-nccl/ (동일 계열 논문 arXiv:2511.09557)
- **all-to-all(EP)**: 노드 간 a2a는 per-peer 소메시지로 쪼개져 α가 곱해지는 구조.
  NCCL 2.12+ PXN이 NIC당 메시지를 최대 8개까지 집약해 2× 이상 개선하나(§2 PXN
  블로그), 노드 간 α 자체(수십 µs)는 남는다 → MoE EP는 노드-로컬 배치가 기본.

## 4. 커널 런치 오버헤드 + vLLM 디코드 스텝 오버헤드

- **커널당 런치 오버헤드 (마이크로벤치, NVIDIA 공식 blog / V100 측정)**:
  커널 실행 2.9 µs짜리를 (a) launch+sync 반복 시 **9.6 µs/kernel** (오버헤드
  ~6.7 µs), (b) 스트림 back-to-back 시 3.8 µs/kernel (오버헤드 ~0.9 µs),
  (c) **CUDA graph 시 3.4 µs/kernel (오버헤드 ~0.5 µs/kernel)**.
  출처: Alan Gray, "Getting Started with CUDA Graphs":
  https://developer.nvidia.com/blog/cuda-graphs/
  (동일 저자 GROMACS 사례: https://developer.nvidia.com/blog/a-guide-to-cuda-graphs-in-gromacs-2023/ ,
  llama.cpp CUDA-graph 적용으로 llama2-7B 전체 10–15% 개선:
  https://github.com/ggml-org/llama.cpp/issues/6763 )
- **LLM 추론 실무 수치 (2차 출처, A100/H100 세대)**: CPU측 커널 launch 5–10 µs,
  디코드 1 forward에 수백 커널 → bs=1에서 launch 오버헤드가 전체 시간의 20–40%,
  CUDA graph가 **토큰당 ~50–100 µs**를 제거.
  출처(2차): https://gigagpu.com/ai-cuda-graphs-vllm/ ,
  https://gigagpu.com/cuda-graph-optimization-inference/ ;
  배경(1차, PyTorch 공식): https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- **vLLM**: 디코드는 기본적으로 CUDA graph 캡처·재생. `--enforce-eager`는 "fastest
  possible startup, at the cost of steady-state decode performance" (공식 문서 원문).
  compile 레벨 -O1=PIECEWISE, -O2=FULL_AND_PIECEWISE cudagraph.
  출처: vLLM docs (stable, 2026-07-28 열람):
  https://docs.vllm.ai/en/stable/configuration/optimization/
- **레이어당 실질 오버헤드 (파생 산출 — K3 93레이어 모델용)**:
  - CUDA graph 有: 커널당 ~0.5 µs × 레이어당 커널 ~5–10개 ≈ **3–5 µs/layer** (런치분).
  - CUDA graph 無(eager): 5–10 µs × 5–10커널 ≈ **25–100 µs/layer** — PP 다노드에서
    graph 캡처가 부분화(PIECEWISE)되면 이 구간이 살아남. 통신 α(§3)는 별도 가산.

## 5. A100 SM80 하드웨어 파라미터 확인

| 항목 | 값 | 확인 출처 |
|---|---|---|
| SM 수 | **108** | Ampere In-Depth blog (§1 링크): "108 SMs" |
| Shared memory/SM | **최대 164 KB** (carveout {0,8,16,32,64,100,132,164} KB 구성 가능; L1+smem 합 192 KB) | CUDA Ampere Tuning Guide: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html ("shared memory capacity per SM is 164 KB", "supports … 0, 8, 16, 32, 64, 100, 132 or 164 KB per SM"); 192 KB 합산은 Ampere In-Depth blog |
| 스레드블록당 최대 dynamic smem | 163 KB (opt-in) | CUDA C Programming Guide, compute capability 8.0 표: https://docs.nvidia.com/cuda/cuda-c-programming-guide/ |
| L2 캐시 | **40 MB** | Tuning Guide ("increases … L2 cache to 40 MB"), Ampere In-Depth blog |
| BF16 dense Tensor Core | **312 TFLOPS** (sparsity 624) — FP16도 312, TF32 156 | A100 80GB datasheet JAN21 (PDF 원문 확인, §1 링크): "Peak BFLOAT16 Tensor Core 312 TF \| 624 TF*" |
| HBM2e 대역 (80GB SXM) | **2,039 GB/s** (40GB SXM=1,555, 80GB PCIe=1,935는 별도 SKU) | A100 80GB datasheet JAN21: "GPU Memory Bandwidth 1,555 GB/s \| 2,039 GB/s" |
| NVLink | 600 GB/s (양방향) | 동일 datasheet Interconnect 항목 |
| TDP (SXM) | 400 W | 동일 datasheet |

주의: Tuning Guide의 "1550 GB/s" 문구는 40GB SKU 기준. 사내 80GB SXM은 2039 GB/s가
맞는 값 (스펙 문서의 이론 하한 0.8 ms/token 산정과 일치).

## 6. 버전/스냅샷 고정

- 실측 NCCL: nccl-tests #149 = NCCL **2.18.1** (컨테이너) / 2.17.1 (호스트), HPC SDK 23.5.
- nccl #949 실측 = 2023년 보고, ConnectX-6 + SR-IOV VM (버전 명시 없음 → 보수측 해석).
- NCCL 2.27 blog = 2025-06 게시. Demystifying NCCL = arXiv:2507.04786v1/v2 (2025-07).
- A100 datasheet = JAN21 리비전(1521051-r2). HGX datasheet = 1485640-r6.
- vLLM docs = stable 채널, 2026-07-28 열람 (vLLM은 주 단위 변동 — Phase 1에서 사내
  반입 시점 커밋으로 재고정할 것).
