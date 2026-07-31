# Kimi K3 — 4×8 A100 5-lane serving 설계

작성일: 2026-07-31
대상 vLLM: `0498dc7ea9204f1863b46515816f5d8d458e01e7`
target checkpoint: `moonshotai/Kimi-K3` 원본 MXFP4
draft checkpoint: `Inferact/Kimi-K3-DSpark` BF16
hardware: 고정 4노드 × 8×A100-80GB(SM80), head 포함 32GPU

## 1. 목적과 판정

하나의 설정을 즉시 최적화하는 대신 다음 위험을 서로 분리한 5개 serving 환경을 만든다.

1. K3 전체 model과 PP 경계의 correctness
2. PP 제거 후 TP/DP/EP 및 cross-node all-to-all correctness
3. DSpark speculative decoding correctness
4. A100에서 1M context를 가능하게 하는 FP8 KV-cache correctness
5. 모든 최적화 조합의 correctness와 성능

각 환경은 별도 폴더, 별도 Compose project/container 이름, 별도 gate stamp 및 로그를
사용한다. 환경은 독립적으로 실행할 수 있지만 32GPU 전체를 사용하므로 동시에 실행하지
않는다.

현재 k01~k07 standalone 결과만으로는 원본 K3 serving이 증명되지 않았다. 기존 k05는
W4A16 Marlin MoE이며 원본 MXFP4가 아니다. 이 설계는 모든 환경을 구현 대상으로 삼되,
해당 환경의 필수 gate가 PASS하기 전 `make up`은 거부하는 gated deployment다.

## 2. 왜 PP를 baseline으로 유지하는가

PP는 필수 topology가 아니라 첫 correctness 기준점이다.

`TP8×PP4`에서는 한 노드가 하나의 PP stage이고, 각 stage 내부는 TP8이다. routed expert도
TP-shard되므로 노드 간 expert all-to-all이 없다. Phase 0 추정 weight footprint는 최악
stage 기준 약 49.6GB/GPU다. 가장 큰 미검증 항목은 AttnRes state와 intermediate tensor가
PP stage 경계를 올바르게 통과하는지다.

`TP8×DP4×EP32`에서는 PP를 제거한다. attention/dense 부분은 각 노드에 TP8 replica로
존재하고 routed experts는 TP×DP=32 ranks 전체에 wide EP로 분산된다. PP/AttnRes 경계는
사라지지만 92개 MoE layer마다 cross-node all-to-all이 생긴다. Phase 0 weight footprint는
약 61.3GB/GPU이며, A100에서 전용 DeepEP를 사용할 수 없어 generic
`allgather_reducescatter`를 최초 correctness backend로 사용한다.

따라서 1번은 model/PP correctness baseline이고 2번 이후가 non-PP optimization lane이다.

## 3. DSpark와 PP/MTP 제약

pinned vLLM의 DSpark loader는 PP group world size가 1이 아니면
`NotImplementedError("DSpark does not support pipeline parallelism.")`을 발생시킨다.
model runner도 DSpark와 PP 조합을 거부한다.

원본 K3 text config의 `num_nextn_predict_layers`는 0이므로 checkpoint 내장 MTP를
대체재로 사용할 수 없다. 따라서 DSpark는 PP가 없는 `TP8×DP4×EP32` 환경에서만
활성화한다.

DSpark는 별도 `Inferact/Kimi-K3-DSpark` checkpoint를 shared HF cache에 준비하고 다음
초기 설정을 사용한다.

```json
{
  "method": "dspark",
  "model": "Inferact/Kimi-K3-DSpark",
  "num_speculative_tokens": 7,
  "attention_backend": "TRITON_MLA",
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "block"
}
```

공개 예시는 `FLASHINFER_MLA`를 사용하지만 A100 correctness lane에서는 patched
`TRITON_MLA`를 사용한다. 이 변경은 k10에서 BF16 reference와 검증한다. DSpark draft는
4B BF16 weights와 추가 cache/workspace를 사용하므로 별도 memory gate가 필요하다.

## 4. 1M context와 k09가 필요한 이유

`TP8×DP4×EP32`의 Phase 0 메모리 추정은 다음과 같다.

| 항목 | GPU당 decimal GB |
|---|---:|
| weights | 61.25 |
| 256k BF16 MLA KV + KDA state, seq 1 | 7.30 |
| 1M BF16 MLA KV + KDA state, seq 1 | 29.05 |
| weights + 1M BF16 state | 90.30 |

A100-80GB는 약 85.9 decimal GB이므로 BF16 KV로는 framework workspace를 제외하고도
1M sequence 하나가 물리적으로 들어가지 않는다.

FP8 cache는 MLA KV 부분을 대략 절반으로 줄여 weights와 1M state 합을 약 75.8GB로
낮춘다. 그러나 pinned `TRITON_MLA`는 FP8 KV 사용 시 SM89+를 요구하고 SM80에서
초기화를 거부한다. K3 MLA는 DCP/PCP도 source assertion으로 거부하므로 context
parallel로 회피할 수 없다.

`k09_mla_fp8_kv_sm80`은 이 간극을 메우는 신규 kernel/backend gate다. A100에 native FP8
Tensor Core 연산을 추가한다는 뜻이 아니다. E4M3 KV를 byte storage로 보관하고
cache insert/load 시 Triton에서 BF16과 software convert하는 correctness-first 경로다.
실제 구현 전 micro-design에서 LUT/bitwise conversion과 scale 단위를 비교해 하나를
선택한다.

## 5. 5개 환경

### 01 `tp8-pp4-256k-safe`

- TP=8, PP=4, DP=1, EP disabled
- `max_model_len=262144`
- `max_num_seqs=1`
- BF16 KV, eager execution
- 목적: 원본 MXFP4 model load, KDA, AttnRes PP crossing, one-token/OpenAI correctness
- 필수 gate: k08 TP-sharded MXFP4, KDA patch, dummy model, TP8×PP4 forward

### 02 `tp8-dp4-ep32-256k-safe`

- TP=8, PP=1, DP=4, `enable_expert_parallel=true`
- effective EP group=32
- `all2all_backend=allgather_reducescatter`
- `max_model_len=262144`
- `max_num_seqs=1` per DP engine
- BF16 KV, eager execution
- 목적: PP 없이 wide EP, internal DP load balancing 및 cross-node communication 검증
- 필수 gate: 01 model gate, k08 EP32 arm, c01 EP communication, one-token parity

### 03 `tp8-dp4-ep32-256k-dspark`

- 02와 동일한 target topology/context
- `Inferact/Kimi-K3-DSpark`, speculative tokens=7
- 목적: bounded context에서 target+draft 통합을 먼저 검증
- 필수 gate: 02 PASS, k10 DSpark SM80, memory profile, greedy/probabilistic parity

### 04 `tp8-dp4-ep32-1m-fp8kv`

- 02 topology
- `max_model_len=1048576`
- SM80 software FP8 MLA KV path
- speculative decoding disabled
- 목적: 1M context와 conservative maximum concurrency 검증
- 필수 gate: 02 PASS, k09 PASS, 1M progressive context smoke

`max_num_seqs`는 임의 상수가 아니라 사내 A100에서 calibration으로 확정한다. 안전 기준은
각 DP engine의 모든 active sequence가 1M까지 성장해도 OOM이 나지 않는 값이다. calibration은
1부터 시작해 증가하며 peak allocated/reserved memory, external workspace, NCCL buffer를
포함하고 최소 5% free-HBM margin을 요구한다. Compose의 최초 값은 1이고, calibration
결과와 image digest가 일치하는 stamp가 있어야 증가시킬 수 있다.

### 05 `tp8-dp4-ep32-1m-fp8kv-dspark`

- 04 target topology/cache
- 03 DSpark 추가
- 목적: 모든 기능을 결합한 최종 최적화 candidate
- 필수 gate: 03과 04 모두 PASS, k09+k10 combined test, 1M target/draft memory profile

05는 DSpark draft weight/cache가 추가되므로 04의 `max_num_seqs`를 그대로 상속하지 않는다.
별도 calibration으로 더 낮은 값을 선택할 수 있다.

## 6. 남은 standalone 및 communication gate

### k08 `mxfp4_marlin_moe_sm80`

원본 K3 MXFP4 routed experts를 위한 가장 선행 gate다.

- MXFP4 packed weights와 E8M0 scales
- K3 latent/intermediate/expert/top-k 실제 shape
- TP-sharded arm과 EP32-sharded arm
- token count 1, 8, 32 및 representative prefill
- unpacked BF16 reference 대비 numeric tolerance
- canary/shape checks
- compute-sanitizer Invalid/Illegal read/write 0건
- allocator OOM과 memory violation 구분

기존 k05 W4A16 PASS는 k08을 대체하지 않는다.

### k09 `mla_fp8_kv_sm80`

- E4M3 byte pack/unpack 및 scale 적용
- K3 cache insert, prefill load, decode load
- BF16 query + software FP8 dequant
- BF16 cache reference 대비 output/error
- sequence lengths 8k, 256k, 1M
- paged block boundary, non-contiguous block table, partial final block
- sanitizer/canary
- 실제 HBM footprint가 BF16 대비 45% 이상 감소
- 1M seq=1에서 runtime headroom 5% 이상

### k10 `k3_dspark_sm80`

- K3 target auxiliary hidden states layers 2/23/47/71/89
- 5-layer non-causal MLA draft, block size 7
- Markov/confidence head
- TRITON_MLA draft cache path
- greedy exactness와 probabilistic rejection correctness
- accepted-token 수 0~7에 따른 KDA/conv state rollback/update
- 256k 및 1M cache/block boundary
- sanitizer/canary와 target-only 대비 memory delta

### c01 `ep32_a2a`

c01은 CUDA kernel gate가 아니라 4-node communication/integration gate다.

- 32 ranks, TP8×DP4 process group construction
- generic allgather/reducescatter correctness
- routed token counts 1, 8, 32 및 skewed expert distribution
- timeout, rank loss, IB/TCP fallback 분류
- node-local 및 cross-node bandwidth/latency 기록
- DeepEP/SM90 backend를 잘못 선택하지 않았는지 assertion

## 7. 폴더 구조와 독립성

```text
deploy/
  01-tp8-pp4-256k-safe/
    compose.yaml
    Makefile
    RUNBOOK.md
    scripts/
  02-tp8-dp4-ep32-256k-safe/
    compose.yaml
    Makefile
    RUNBOOK.md
    scripts/
  03-tp8-dp4-ep32-256k-dspark/
    compose.yaml
    Makefile
    RUNBOOK.md
    scripts/
  04-tp8-dp4-ep32-1m-fp8kv/
    compose.yaml
    Makefile
    RUNBOOK.md
    scripts/
  05-tp8-dp4-ep32-1m-fp8kv-dspark/
    compose.yaml
    Makefile
    RUNBOOK.md
    scripts/
  tooling/
    render_lanes.py
    validate_lanes.py
```

각 lane 폴더는 그 폴더만 보고 `make preflight/up/down/status/logs/smoke/gates`를 실행할 수
있도록 self-contained 산출물을 커밋한다. 반복 파일은 `render_lanes.py`로 생성하되 생성된
Compose/Makefile/scripts도 repository에 커밋한다. 따라서 사내 반입 후 generator 없이도
실행 가능하고, CI는 재생성 diff가 0인지 검사해 drift를 막는다.

각 Compose는 하나의 파일에 `head`, `worker1`, `worker2`, `worker3` profile을 모두
포함한다. `.env`와 `${VAR}` interpolation을 사용하지 않는다. node IP, SSH host/user,
NCCL interface 등 현장값은 각 Compose에 `CHANGE_ME_*` literal로 기록하고 preflight가
잔존 sentinel을 거부한다.

shared paths:

- HF cache: `/mnt/nas/home/models` read-only
- serving root: `/mnt/nas/home/ubuntu/service/large_model/Kimi-K3`
- lane logs/stamps: `<serving-root>/runs/<lane-id>/`

각 lane은 고유 Compose project/container name, API port, rendezvous port, DP RPC port를
사용한다. NFS global lock으로 다른 lane이 32GPU를 사용 중이면 `make up`을 거부한다.

## 8. 공통 orchestration

모든 lane Makefile은 head에서 실행한다.

- `make preflight`: sentinel, image digest, 8×A100/node, shared cache revision,
  ports/interfaces, lane-specific stamps 확인
- `make gates`: 필요한 k08/k09/k10/c01 및 integration gate 실행
- `make up`: head profile 시작 후 SSH로 worker profiles 병렬 시작
- `make status`: 4노드 process/rank/GPU/API와 DP/EP group 상태
- `make logs`: NFS structured logs 시간순 출력
- `make smoke`: lane별 context/speculative test
- `make calibrate`: 04/05에서만 concurrency 탐색
- `make down`: worker와 head를 idempotent하게 정리하고 NFS lock 해제

native multi-node MP를 사용한다. world size는 다음과 같다.

- 01: TP8×PP4×DP1=32
- 02~05: TP8×PP1×DP4=32

Ray와 external launcher는 사용하지 않는다.

## 9. lane별 RUNBOOK 계약

각 `RUNBOOK.md`는 다음 항목을 같은 순서로 포함한다.

1. 이 lane이 검증하는 것과 검증하지 않는 것
2. topology와 GPU/rank mapping
3. 사내 AI가 채워야 할 `CHANGE_ME_*` 목록
4. 필요한 model/draft snapshot revision과 offline cache 확인법
5. image 반입/load/digest 확인
6. `make preflight`
7. standalone/communication/integration gate 실행 순서
8. `make up`, readiness 판단, OpenAI request
9. numeric/context/speculative PASS 기준
10. 예상 오류 문자열별 다음 진단 명령
11. 로그에서 사용자가 눈으로 전달할 최소 필드
12. `make down` 및 잔존 process/GPU allocation 확인
13. 다음 lane으로 승격하는 조건

RUNBOOK은 tarball 회신을 요구하지 않는다. 각 실패 단계는 한 줄의 stable error code와
최대 20줄의 요약을 출력해 사용자가 텍스트로 전달할 수 있게 한다.

## 10. 통합 승격 순서와 PASS

```text
k08 + KDA patch
       |
       v
01 TP/PP 256k
       |
       v
c01 + k08 EP arm --> 02 TP/DP/EP 256k
                         |             |
                         v             v
                   k10 --> 03      k09 --> 04
                         |             |
                         +------> 05 <-+
```

01 PASS:

- 원본 checkpoint load/profile
- TP8×PP4 one-token forward
- 8k/128k/256k context smoke
- OpenAI response와 single-process reference의 token/logit tolerance

02 PASS:

- EP32 dispatch/combine numeric correctness
- 4 DP engines readiness 및 routing
- 256k seq=1
- 01 대비 greedy output parity

03 PASS:

- target-only와 DSpark output distribution/greedy parity
- acceptance length와 fallback rate 기록
- accepted-token 모든 경우 state correctness
- target-only보다 decode latency가 악화되면 기능 PASS/성능 FAIL을 분리

04 PASS:

- 1M prompt progressive checkpoints: 256k, 512k, 768k, 1M
- retrieval/logit parity
- FP8 KV memory reduction과 free-HBM margin
- calibrated worst-case `max_num_seqs`

05 PASS:

- 04 context gate와 03 speculative gate를 동시에 만족
- combined memory calibration
- target-only 04 대비 실제 decode speedup
- acceptance가 낮은 workload에서 자동 fallback correctness

## 11. 구현 순서

모든 lane 폴더는 먼저 생성하지만 gate 구현과 실제 승격은 다음 순서를 따른다.

1. common image/patch provenance
2. k08
3. lane 01
4. c01 및 k08 EP arm
5. lane 02
6. k10과 lane 03
7. k09과 lane 04
8. combined k09+k10과 lane 05

즉 5개 폴더와 runbook은 한 번에 준비하되, 앞 단계 실패를 뒤 단계가 가리지 않도록
실행 dependency를 유지한다.

## 12. 완료 판정

구현 완료는 파일 생성만을 뜻하지 않는다.

1. 5개 lane 폴더가 각자 Compose/Makefile/RUNBOOK/scripts를 포함
2. 각 Compose에 4개 node profile이 있고 별도 env 파일이 없음
3. static/schema/unit test 및 generator drift test PASS
4. host CPU dry-run이 GPU 실행을 PASS로 사칭하지 않음
5. 사내 A100에서 lane별 필수 gate 결과가 stable error code로 기록됨
6. 01~05가 앞 절의 lane PASS 기준을 각각 충족
7. 한 lane의 실패/로그/stamp가 다른 lane과 섞이지 않음
8. `make down` 후 4노드에 container/process/GPU allocation이 남지 않음

현재 개발 호스트에는 A100이 없으므로 1~4의 정적 완료와 GPU 실행용 정확한 절차를
제공할 수 있다. 5~8의 최종 PASS는 사내 A100 실행 결과로만 확정한다.
