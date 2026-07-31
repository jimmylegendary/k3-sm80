# Kimi K3 — 4×8 A100 multi-node MP serving 설계

> 이 문서는 최초 단일환경 설계 기록이다. 5개 독립환경과 추가 kernel gate를 포함한 최신
> 설계는
> [`2026-07-31-kimi-k3-5lane-serving-design.md`](2026-07-31-kimi-k3-5lane-serving-design.md)를
> 따른다.

작성일: 2026-07-31
대상 vLLM: `0498dc7ea9204f1863b46515816f5d8d458e01e7`
대상 checkpoint: 원본 `moonshotai/Kimi-K3` MXFP4
대상 하드웨어: 고정 4노드 × 8×A100-80GB(SM80), head 포함 총 32GPU

## 1. 목적과 현재 판정

이 설계의 목적은 Kimi K3 원본 MXFP4 checkpoint를 4개 A100 노드에서 vLLM의
OpenAI-compatible API로 기동·중지·관찰하는 재현 가능한 배포 단위를 만드는 것이다.

현재 커널 단독 검증은 KDA prefill/decode, causal conv, MLA decode, SiTU, Marlin W4A16
MoE 경로의 SM80 실행 가능성을 확인했다. KDA chunk의 128k int32 offset 문제는
`kernel-kit/vendor/fla_kda/chunk.py`에서 수정됐다. 그러나 다음 항목은 아직 전체 서빙
성립을 증명하지 않았으므로 배포 시작 전 hard gate로 둔다.

1. 원본 K3 형상의 MXFP4 Marlin MoE를 A100에서 단독 수치·shape·sanitizer 검증
2. KDA int64 수정이 실제 vLLM 설치 패키지에 반영됐는지 검증
3. K3 축소 config + dummy weight 전체 model forward
4. 원본 checkpoint의 4노드 load/profile run
5. TP8×PP4에서 AttnRes state가 PP 경계를 통과하는지 확인

따라서 산출물은 “이미 검증된 production deployment”가 아니라, 위 gate를 순서대로
통과시키면서 실제 서빙까지 승격할 수 있는 **gated deployment**다.

## 2. 확정된 사용자 계약

- distributed backend: vLLM native multi-node `mp`; Ray를 사용하지 않는다.
- 고정 topology: 4노드, 노드당 A100 8장.
- head(node rank 0)도 8GPU를 모델 연산에 사용하며 OpenAI API를 노출한다.
- 병렬화 기본값: tensor parallel 8, pipeline parallel 4.
- checkpoint lane: 원본 MXFP4. W4A16은 진단/비상 fallback으로만 유지한다.
- Hugging Face cache: 모든 노드에서 동일한 NFS 경로 `/mnt/nas/home/models`.
- serving 배포 루트: 모든 노드에서 동일한 NFS 경로
  `/mnt/nas/home/ubuntu/service/large_model/Kimi-K3`.
- 모델 식별자는 원본 `moonshotai/Kimi-K3`로 고정하고, 네 노드 모두 shared cache를
  offline mode로 해석한다. 실제 snapshot 존재 여부는 preflight가 확인한다.
- Compose는 하나만 사용한다. 별도 `.env`와 노드별 Compose 파일을 만들지 않는다.
- 노드별 고정값은 단일 Compose 파일의 해당 service/profile에 직접 기록한다.
- profile은 `head`, `worker1`, `worker2`, `worker3`으로 구분한다.
- head에서 SSH로 worker 3개 노드에 접속할 수 있으며, SSH key는 이미 등록돼 있다.
- 미확정 현장값은 `CHANGE_ME_*` sentinel로 표시한다. 현장 AI가 단일 Compose 파일을
  수정하고 `make preflight`가 sentinel 잔존을 거부한다.

## 3. MP 선택 근거

pinned vLLM의 `ParallelConfig`는 다음 multi-node MP 필드를 직접 제공한다.

- `distributed_executor_backend="mp"`
- `master_addr`
- `master_port`
- `node_rank`
- `nnodes`

`world_size = tensor_parallel_size × pipeline_parallel_size = 32`이며
`local_world_size = world_size / nnodes = 8`이다. 따라서 각 노드는 local rank 0~7을
생성하고 node rank에 따라 global rank를 배정한다.

Ray는 동적 placement가 필요 없는 고정 4노드에 불필요한 control plane과 운영 포트를
추가하므로 제외한다. `external_launcher`/torchrun은 pinned vLLM에서 pipeline
parallelism과 함께 사용할 수 없으므로 제외한다.

## 4. 전체 구조

```text
                       OpenAI-compatible API :8000
                                  │
                       head / node rank 0
                       8×A100, PP stage 0
                       MP rendezvous :29501
                         ╱       │       ╲
               SSH + NCCL   SSH + NCCL   SSH + NCCL
                   ╱            │             ╲
          worker1/rank1   worker2/rank2   worker3/rank3
          8×A100/stage1   8×A100/stage2   8×A100/stage3
                   ╲            │             ╱
           shared NFS HF cache + shared serving directory
```

모든 컨테이너는 `network_mode: host`, `ipc: host`, 동일 이미지와
`/mnt/nas/home/models` HF cache를 사용한다. Compose·Makefile·로그는 모든 노드가 동일하게
보는 `/mnt/nas/home/ubuntu/service/large_model/Kimi-K3` 아래에 둔다. 노드 간 NCCL/Gloo
연결은 Compose에 기록된 node IP와 interface를 사용한다.

## 5. 산출물 구조

```text
deploy/
  compose.yaml
  Makefile
  RUNBOOK.md
  scripts/
    compose_config.py
    preflight.sh
    remote.sh
    wait_cluster.sh
    smoke_openai.sh
docker/
  Dockerfile.openai
  patches/
    kda-sm80-int64.patch
  build-openai.sh
  push-openai.sh
kernel-kit/
  k08_mxfp4_marlin_moe/
    harness.py
tests/
  deploy/
    test_compose_config.py
    test_make_dry_run.sh
    test_patch_applied.sh
```

## 6. Serving 이미지

공식 `vllm/vllm-openai` 이미지를 그대로 사용하지 않는다. 현재 공식 이미지는 이 설계가
고정한 K3 branch, SM80 build, KDA 수정의 조합을 보장하지 않기 때문이다.

`Dockerfile.openai`는 다음 계약을 갖는다.

1. vLLM `0498dc7e` source를 가져온다.
2. `kda-sm80-int64.patch`를 source tree의 실제 KDA 파일에 적용한다.
3. `TORCH_CUDA_ARCH_LIST=8.0`으로 Marlin/CUDA extension을 빌드한다.
4. 빌드 결과에 원본 MXFP4/Marlin 코드와 patched Triton KDA 코드가 모두 포함됐는지
   build-time assertion으로 확인한다.
5. OpenAI-compatible `vllm serve` CLI와 MP runtime 의존성을 포함한다.
6. image label에 vLLM commit, patch hash, CUDA, torch, SM target을 기록한다.
7. 사내 registry가 차단돼 있으므로 기존 방식과 동일하게 GitHub Release용
   `docker save | zstd | split` 산출물을 만든다.

이미지 이름은 로컬 load 후 사용할 고정 별칭
`k3-sm80-openai:0498dc7e-sm80`로 한다. Compose는 GHCR 주소가 아니라 이 로컬 별칭을
참조한다.

## 7. 단일 Compose 파일

repository의 `deploy/` 산출물은 사내 반입 시
`/mnt/nas/home/ubuntu/service/large_model/Kimi-K3`에 배치한다.
`deploy/compose.yaml`에는 네 service가 모두 들어간다.

| service | profile | node rank | 역할 |
|---|---|---:|---|
| `k3-head` | `head` | 0 | MP leader, PP stage 0, API |
| `k3-worker1` | `worker1` | 1 | PP stage 1 |
| `k3-worker2` | `worker2` | 2 | PP stage 2 |
| `k3-worker3` | `worker3` | 3 | PP stage 3 |

공통 설정은 YAML anchor로 재사용하되 node IP, SSH host/user, node rank, network
interface는 각 service에 직접 적는다. Makefile이 읽어야 할 orchestration 정보는 service
label에도 기록한다.

```yaml
labels:
  io.k3.node-rank: "1"
  io.k3.ssh-host: "CHANGE_ME_WORKER1_SSH_HOST"
  io.k3.ssh-user: "CHANGE_ME_WORKER1_SSH_USER"
  io.k3.node-ip: "CHANGE_ME_WORKER1_NODE_IP"
```

별도 `.env`를 사용하지 않으며 `${VAR}` interpolation에 의존하지 않는다.
각 service의 `environment`에는 `HF_HOME: /mnt/nas/home/models`와
`HF_HUB_OFFLINE: "1"`을 literal로 기록한다.

각 service의 핵심 vLLM 인자는 다음과 같다.

```text
vllm serve moonshotai/Kimi-K3
  --distributed-executor-backend mp
  --nnodes 4
  --node-rank <0..3>
  --master-addr <head node IP>
  --master-port <port>
  --tensor-parallel-size 8
  --pipeline-parallel-size 4
  --quantization mxfp4
  --dtype bfloat16
  --max-model-len 262144
  --max-num-seqs 1
  --trust-remote-code
  --enforce-eager
```

초기 correctness lane은 BF16 KV cache와 eager execution을 사용한다. FP8 KV, CUDA graph,
larger batch는 첫 OpenAI smoke 및 logits parity 이후 별도 tuning 단계에서만 활성화한다.
KDA prefill backend는 patched Triton 경로를 명시적으로 선택한다. 정확한 CLI key는
image build 시 pinned K3 config schema에서 추출해 test로 고정한다.

## 8. Makefile orchestration

Makefile은 head에서만 실행한다.

### `make preflight`

다음을 로컬과 SSH worker에서 확인하고 하나라도 실패하면 시작하지 않는다.

- Compose의 `CHANGE_ME_*` sentinel이 0개
- 네 service의 rank가 정확히 0,1,2,3이고 IP/SSH host가 중복되지 않음
- 모든 노드에서 이미지 digest가 동일
- 모든 노드에서 A100 8장, compute capability 8.0, driver 호환
- 모든 노드에서 `/mnt/nas/home/models`의 동일 K3 snapshot revision이 resolve되고
  config/index/모든 shard가 읽힘
- 모든 노드에서 `/mnt/nas/home/ubuntu/service/large_model/Kimi-K3`가 동일 NFS
  filesystem으로 보이고 Compose/Makefile revision이 동일
- 모든 노드에서 head master IP/port 연결 가능
- 선택한 NCCL/Gloo interface와 `/dev/infiniband` 존재 여부
- MXFP4 gate stamp가 현재 image digest 및 repository commit과 일치
- KDA patch-applied assertion 및 K3 dummy gate stamp 존재

### `make up`

1. preflight 실행
2. head profile을 detached mode로 시작
3. worker1~3에 SSH 접속해
   `/mnt/nas/home/ubuntu/service/large_model/Kimi-K3`에서 해당 profile을 병렬 시작
4. MP rendezvous 및 32 global ranks 준비를 기다림
5. head API `/health` 준비를 기다림
6. 최소 OpenAI `/v1/models` 및 `/v1/chat/completions` smoke 실행

중간 실패 시 시작된 worker와 head를 역순으로 정리하고 실패 노드/단계를 출력한다.

### 기타 target

- `make down`: head API 종료 후 worker3~1, head 순으로 종료
- `make restart`: `down` 후 `up`
- `make status`: 네 노드 container/process/GPU/rank/API 상태를 표로 출력
- `make logs`: head 및 worker 로그의 시간순 tail
- `make logs-head`, `make logs-worker1` 등 노드별 로그
- `make smoke`: health, model listing, 짧은 completion
- `make gate-mxfp4`: k08 standalone gate
- `make gate-k3-dummy`: patched image의 축소 K3 dummy forward
- `make gate-load`: 실제 checkpoint load/profile gate
- `make config`: Compose를 canonical JSON으로 렌더링하고 node table 출력

## 9. MXFP4 hard gate

기존 k05는 W4A16 Marlin MoE를 검증했으므로 원본 K3 MXFP4의 증거가 아니다.
`k08_mxfp4_marlin_moe`를 별도로 추가한다.

필수 판정:

- K3 routed expert 실제 shape: latent 3584, intermediate 3072, top-k 16
- MXFP4 packed weight + E8M0 scale를 원본 backend와 동일하게 구성
- token counts 1, 8, 32 및 prefill representative
- BF16 reference 대비 numeric tolerance
- shape/canary coverage
- compute-sanitizer에서 실제 Invalid/Illegal read/write 0건
- `cudaErrorMemoryAllocation` allocator retry는 memory violation과 구분
- 측정값은 기록하되 correctness gate와 분리

PASS 결과는 image digest, GPU UUID/driver, vLLM commit, harness commit을 포함한 stamp로 NFS에
저장한다. stamp가 현재 배포와 다르면 `make up`이 거부한다.

## 10. vLLM 통합 gate

순서는 반드시 다음과 같다.

1. image 내부 KDA 파일에 int64 patch가 적용됐는지 source assertion
2. vendored standalone KDA와 image 내부 vLLM KDA의 source hash/필수 diff 검증
3. 축소 K3 config + dummy weights로 KDA/AttnRes/LatentMoE/MLA forward
4. 단일 노드 8GPU에서 model initialization 및 profile run
5. 4노드에서 checkpoint load
6. TP8×PP4 one-token forward
7. OpenAI endpoint 1-token completion
8. 8k, 128k, 256k context 단계적 smoke

각 단계는 이전 단계 PASS 없이 실행할 수 없게 Make target dependency로 연결한다.

## 11. 로그와 실패 처리

- 모든 service는 JSON-compatible timestamped log를
  `/mnt/nas/home/ubuntu/service/large_model/Kimi-K3/logs`에 쓴다.
- 파일명은 run id, node rank, image digest를 포함한다.
- `make logs`는 SSH stdout이 아니라 NFS log를 우선 사용하고, container tail을 보조로 쓴다.
- head/worker의 exit code, OOM, NCCL timeout, rank join 실패를 서로 구분한다.
- startup timeout 시 현재 접속 rank, GPU count, 마지막 NCCL/vLLM 오류를 압축 출력한다.
- `make down`은 idempotent하며 일부 노드 SSH 실패가 있어도 나머지 노드 정리를 계속한다.
- 강제 삭제/데이터 삭제를 수행하지 않는다. checkpoint와 HF cache는 read-only mount한다.

## 12. 네트워크 및 컨테이너 안전 설정

- `network_mode: host`
- `ipc: host`
- GPU 8장 노출
- `ulimits.memlock = -1`, 충분한 stack
- `/mnt/nas/home/models` HF cache mount는 read-only
- `/dev/infiniband`는 현장 preflight에서 존재할 때만 활성화하도록 명확한 Compose block으로 둔다.
- `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA`,
  `NCCL_IB_GID_INDEX`는 각 service에 직접 기록하며 sentinel 검사를 받는다.
- SSH host key 검사를 비활성화하지 않는다.
- API bind 및 접근 제어 값도 Compose에 직접 기록한다. 기본 설계는 사내망 bind이며 외부 공개를
  전제로 하지 않는다.

## 13. 테스트 전략

### 개발 호스트

- Compose schema 및 profile rendering
- sentinel 검출
- service rank/IP uniqueness
- Makefile dry-run과 SSH command snapshot
- image label/source patch assertion
- shell syntax 및 Python unit tests
- Docker GPU가 없는 환경에서는 GPU gate를 SKIP이 아니라 명시적 “not executed”로 기록

### 사내 A100

- `make preflight`
- `make gate-mxfp4`
- `make gate-k3-dummy`
- `make gate-load`
- `make up`
- `make smoke`
- `make status`
- `make down`

## 14. 완료 판정

배포 작업은 다음을 모두 만족해야 완료다.

1. 단일 Compose에서 네 profile이 올바른 rank/host로 렌더링됨
2. 별도 env 없이 Compose 하나를 현장 AI가 수정할 수 있음
3. head의 Makefile이 SSH로 worker 3개 profile을 시작·중지·관찰함
4. 모든 노드가 동일 image digest와 동일 HF cache snapshot revision을 사용함
5. MXFP4 standalone gate PASS
6. patched K3 dummy gate PASS
7. 실제 K3 checkpoint의 4노드 load/profile PASS
8. TP8×PP4 OpenAI completion이 정상 token을 반환
9. `make down` 후 네 노드에 배포 container가 남지 않음
10. runbook만 보고 현장 AI가 sentinel을 채우고 동일 절차를 재현함

성능 최적화는 위 correctness 완료 뒤 별도 단계다. 현재 측정한 A100 HBM 1727GB/s와
커널별 달성 효율을 Phase 0 모델에 재입력해 목표 tok/s를 갱신하고, 그 후 eager 해제,
CUDA graph, KV dtype, batch 및 통신 tuning을 진행한다.
