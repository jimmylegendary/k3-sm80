# 11-k31-tp8-pp4-256k-safe

## 1. 검증 범위

이 lane은 원본 `moonshotai/Kimi-K3.1-Pro` MXFP4 checkpoint를 SM80에서
TP8/PP4/DP1, EP=0로 기동하는 단계를 검증한다. DSpark=0,
software FP8 KV=0다. 필수 최종 gate는 `k08 kda k11 k12 pp`다.
사내 A100에서 stamp가 만들어지기 전 상태는 `INCOMPLETE`이며, host dry-run은 PASS가 아니다.

P1 상한: K3.1-Pro는 DSA(DeepSeek Sparse Attention) indexer가 붙은 24개 attention 층이
SM80에서 vLLM 기본 경로(DeepGEMM, FlashMLA sparse)로 돌지 않는다. 이 lane은 SM80 전용 경로
(indexer logits k11, sparse MLA 백엔드 `TRITON_MLA_SPARSE` k12)를 이미지 패치로 얹어 검증한다.
속도는 보증하지 않으며(kernel PERF는 측정 전용), 256k, BF16 KV, `max_num_seqs=1`,
`enforce_eager`, DSpark 없음, 1M 미지원이다. indexer Q는 FP8로 양자화하지 않아 DeepGEMM 대비
top-k 경계가 달라질 수 있다. 근거는 [K31-SUPPORT-REVIEW.md](/home/jimmy/repos/k3-sm80/docs/K31-SUPPORT-REVIEW.md) §2, §5다.

## 2. 토폴로지와 rank

| node/profile | node rank | local GPU | 역할 |
|---|---:|---:|---|
| head | 0 | 0–7 | API + native MP launcher |
| worker1 | 1 | 0–7 | native MP `--headless` worker |
| worker2 | 2 | 0–7 | native MP `--headless` worker |
| worker3 | 3 | 0–7 | native MP `--headless` worker |

world size는 TP8 × PP4 × DP1 = 32다. Ray를 쓰지 않고
`--distributed-executor-backend mp --data-parallel-backend mp --nnodes 4`를
명시하여 실행기와 DP 제어면을 모두 native MP로 고정한다.
node rank 0만 OpenAI API를 열고, rank 1–3은 `--headless`로 engine만
실행한다.

## 3. 단일 `.env`

이 폴더의 `.env` 하나만 네 노드가 공통 사용한다. 먼저 아래 placeholder를 사내 값으로
채운다.

```bash
grep '^K3_.*CHANGE_ME' .env
```

head/worker 4개의 IP·SSH host·SSH user, `K3_NCCL_SOCKET_IFNAME`,
`K3_GLOO_SOCKET_IFNAME`, `K3_NCCL_IB_HCA`가 0건 남아야 한다. topology,
MXFP4/Marlin, KDA Triton, KV dtype, context, concurrency, DSpark 옵션은 모두 같은 `.env`의
`VLLM_*` 항목으로 제어한다. 값을 바꾸면 기존 stamp는 env SHA 불일치로 자동 폐기된다.
K3.1 DSA 층(24개)의 MLA KV cache는 BF16이며 `--kv-cache-dtype bfloat16`을 명시한다. `auto`/`fp8*`는 vLLM `KimiK3ForConditionalGenerationConfig`가 `fp8_ds_mla`로 재작성하므로 쓰지 않는다. DSA indexer-K cache는 vLLM 기본 형식(E4M3 128 B + fp32 scale 4 B)이고 SM80 Triton MQA logits가 읽는다. software FP8 KV 패치는 이 lane에서 적용하지 않는다.

## 4. 모델 cache

공유 HF cache는 `/mnt/nas/home/models`, service/repository root는
`/mnt/nas/home/ubuntu/service/large_model/Kimi-K3.1-Pro`다. 네 노드에서 같은 snapshot을
보는지 확인한다.

```bash
find /mnt/nas/home/models -path '*models--moonshotai--Kimi-K3.1-Pro/snapshots/*' -maxdepth 5 -type d
```

DSpark=0이면 draft cache는 필수 조건이 아니다. 인터넷 없이 실행하므로 누락 snapshot을
serving 중 내려받으려 하지 않는다.

## 5. 이미지

`.env`의 `K3_IMAGE`를 네 노드에 같은 digest로 load/pull한 뒤 head에서 확인한다.

```bash
docker image inspect --format '{{.Id}} {{index .Config.Labels "io.k3.patchset"}}' "$(sed -n 's/^K3_IMAGE=//p' .env)"
python3 ../../docker/verify_k31_image.py "$(sed -n 's/^K3_IMAGE=//p' .env)" --gpu
```

이 lane의 이미지는 K3 이미지(vLLM 핀 `0498dc7e`)가 아니라 K3.1 이미지(vLLM main 핀
`a3c14efb`)다. `io.k3.patchset` 라벨은 K3.1 패치 세트(`docker/patches-k31/`)를 가리켜야 하며,
K3 patchset(`kda-sm80-int64` 등)이 보이면 잘못된 이미지다. 두 이미지를 섞어 배포하지 않는다.

## 6. Preflight

```bash
make config
make preflight
```

정상 종료는 `K3-PREFLIGHT-PASS ... nodes=4 gpu=32xA100 image=sha256:...` 한 줄이다.
이 단계는 Compose schema, NFS sentinel, model cache, 네 노드 image digest와 각 8×A100을
검증한다. 기존 `k3-*` container나 CUDA compute process가 하나라도 있으면
`PREFLIGHT-K3-CONTAINER-BUSY`/`PREFLIGHT-GPU-BUSY`로 중단한다.

K3 lane(01–05)과 이 lane은 serving root가 달라 32-GPU cluster lock
(`<serving-root>/runs/.k3-32gpu.lock`)을 서로 공유하지 않는다. 한쪽 lane의 service나 gate가
실행 중일 때 다른 쪽을 시작하지 않는다. `make gates`와 `make up`은 먼저 preflight를 거쳐
막히지만, `make gate-k11` 같은 단독 gate 실행은 preflight를 거치지 않아 막지 못한다.

## 7. Gate 실행

service를 띄우기 전에 다음을 실행한다.

```bash
make gates
```

이 lane의 선행 gate는 `k08 kda k11 k12`다. 각 gate는 repo commit, image digest,
`.env` SHA에 묶인 NFS stamp를 남긴다. `K3-GATE-PASS`가 아닌
`K3-GATE-FAIL/INCOMPLETE/VERIFY-FAIL`이면 serving으로 넘어가지 않는다.

k11(`k11_dsa_indexer_sm80.json`)과 k12(`k12_sparse_mla_sm80.json`)는 kernel-kit 하네스를
서빙 이미지 안에서 실행하는 SM80 DSA kernel gate이며 `make gate-k11`, `make gate-k12`로 개별
실행할 수 있다. PASS는 NUMERIC, SANITIZER, SHAPES 기준이고 PERF는 측정 전용이다.

## 8. Serving 실행

```bash
make up
make gate-load
make gate-context
make smoke
make status
```

context checkpoint는 `8192 → 131072 → 262144` 순서다. 이 lane은 다음 DSpark 비교용 baseline fixture를 만들지 않는다.
이 lane에는 DSpark gate를 적용하지 않는다. 이 lane에는 calibration을 적용하지 않는다.

`make smoke`는 `/v1/completions` greedy 요청 2회의 출력 동일성과 `/v1/chat/completions`
`reasoning_effort: "high"` 요청(reasoning 채널이 비어 있지 않고 2회 동일)을 함께 검증한다.

`make up`은 head profile을 먼저 시작하고 SSH로 worker1–3 profile을 병렬 시작한다.
`K3-HEALTH PASS ... expected_ranks=32`와 `K3-UP-PASS`가 모두 보여야 한다.

## 9. PASS 기준

- load: 원본 MXFP4 checkpoint profile/first token과 32 rank 초기화 PASS
- context: `8192 → 131072 → 262144` 각 요청이 1 token을 생성
- memory: context/calibration 종료 후 모든 GPU free HBM ≥5%
- topology: lane 11은 PP4 stamp가 현재 env와 일치
- smoke: greedy completion 2회 동일, `reasoning_effort: "high"` reasoning 채널 관측
- DSpark=1: 직전 target-only baseline과 greedy output 일치, speculative metric의
  draft 및 accepted position 0–6 관측, draft memory delta 양수
- `make verify-all-gates`가 `K3-GATE-VERIFY-PASS ... scope=all` 출력

## 10. 오류 코드

| stable code | 의미 | 다음 확인 |
|---|---|---|
| `K3-PREFLIGHT-FAIL` | 노드/image/NFS/cache 불일치 | 같은 줄의 `node_rank`와 rc 확인 |
| `K3-GATE-FAIL` | standalone/communication kernel 실패 | 해당 gate log 마지막 20줄 전달 |
| `K3-GATE-VERIFY-FAIL` | stamp 누락 또는 code/image/env 변경 | gate 재실행 |
| `K3-UP-FAIL` | 4 launcher 또는 API readiness 실패 | `make logs`, `make status` |
| `K3-SMOKE-FAIL` | OpenAI/context/memory 검증 실패 | report의 `error` 한 줄 확인 |
| `K3-DOWN-LOCK-FAIL` | 다른 lane lock 또는 손상된 owner | lock owner JSON 확인 후 임의 삭제 금지 |

## 11. 전달할 출력

tgz/tarball은 필요 없다. 실패 시 아래만 텍스트로 전달한다.

```bash
make status 2>&1 | tail -20
make logs 2>&1 | tail -20
```

stable error code, gate/lane, node rank, rc, report path, 오류 전후 최대 20줄만 전달한다.
JSON 전체나 모델 파일은 전송하지 않는다.

## 12. 정리

```bash
make down
make status
nvidia-smi
```

네 profile의 container가 없어야 하고, GPU process가 남지 않아야 하며,
`<serving-root>/runs/.k3-32gpu.lock`도 이 lane owner에 한해 제거되어야 한다.

## 13. 승격

현재 lane의 `make verify-all-gates`, `make smoke`, `make status`가 PASS이고 정리 후
잔존 process가 없을 때만 `K3.1 P2`로 이동한다. 앞 lane FAIL을 뒤 lane 실행으로
덮지 않으며, 실행하지 않은 lane은 FAIL이 아니라 `INCOMPLETE`로 둔다.
K3.1 P2(DP4/EP32 + DSpark preview)와 1M lane은 아직 구현되지 않았다.
