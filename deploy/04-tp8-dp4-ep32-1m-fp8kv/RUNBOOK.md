# 04-tp8-dp4-ep32-1m-fp8kv

## 1. 검증 범위

이 lane은 원본 `moonshotai/Kimi-K3` MXFP4 checkpoint를 SM80에서
TP8/PP1/DP4, EP=1로 기동하는 단계를 검증한다. DSpark=0,
software FP8 KV=1다. 필수 최종 gate는 `k08 c01 k09 ep 1m`다.
사내 A100에서 stamp가 만들어지기 전 상태는 `INCOMPLETE`이며, host dry-run은 PASS가 아니다.

## 2. 토폴로지와 rank

| node/profile | node rank | local GPU | 역할 |
|---|---:|---:|---|
| head | 0 | 0–7 | API + native MP launcher |
| worker1 | 1 | 0–7 | native MP worker |
| worker2 | 2 | 0–7 | native MP worker |
| worker3 | 3 | 0–7 | native MP worker |

world size는 TP8 × PP1 × DP4 = 32다. Ray를 쓰지 않고
`--distributed-executor-backend mp --data-parallel-backend mp --nnodes 4`를
명시하여 실행기와 DP 제어면을 모두 native MP로 고정한다.

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
target MLA cache는 SM80 software FP8 storage를 사용한다. native FP8 arithmetic은 사용하지 않는다.

## 4. 모델 cache

공유 HF cache는 `/mnt/nas/home/models`, service/repository root는
`/mnt/nas/home/ubuntu/service/large_model/Kimi-K3`다. 네 노드에서 같은 snapshot을
보는지 확인한다.

```bash
find /mnt/nas/home/models -path '*models--moonshotai--Kimi-K3/snapshots/*' -maxdepth 5 -type d
find /mnt/nas/home/models -path '*models--Inferact--Kimi-K3-DSpark/snapshots/*' -maxdepth 5 -type d
```

DSpark=0이면 draft cache는 필수 조건이 아니다. 인터넷 없이 실행하므로 누락 snapshot을
serving 중 내려받으려 하지 않는다.

## 5. 이미지

`.env`의 `K3_IMAGE`를 네 노드에 같은 digest로 load/pull한 뒤 head에서 확인한다.

```bash
docker image inspect --format '{{.Id}} {{index .Config.Labels "io.k3.patchset"}}' "$(sed -n 's/^K3_IMAGE=//p' .env)"
python3 ../../docker/verify_image.py "$(sed -n 's/^K3_IMAGE=//p' .env)" --gpu
```

patchset에는 `kda-sm80-int64`, `mxfp4-marlin-moe-sm80-v2`,
`mla-fp8-kv-sm80-v1`, `dspark-bf16-v1`가 포함되어야 한다. software FP8은
lane 04/05에서만 opt-in되고, `dspark-bf16-v1`은 lane 05의 RoPE draft cache를
BF16으로 분리한다.

## 6. Preflight

```bash
make config
make preflight
```

정상 종료는 `K3-PREFLIGHT-PASS ... nodes=4 gpu=32xA100 image=sha256:...` 한 줄이다.
이 단계는 Compose schema, NFS sentinel, model cache, 네 노드 image digest와 각 8×A100을
검증한다. 기존 `k3-*` container나 CUDA compute process가 하나라도 있으면
`PREFLIGHT-K3-CONTAINER-BUSY`/`PREFLIGHT-GPU-BUSY`로 중단한다.

## 7. Gate 실행

service를 띄우기 전에 다음을 실행한다.

```bash
make gates
```

이 lane의 선행 gate는 `k08 c01 k09`다. 각 gate는 repo commit, image digest,
`.env` SHA에 묶인 NFS stamp를 남긴다. `K3-GATE-PASS`가 아닌
`K3-GATE-FAIL/INCOMPLETE/VERIFY-FAIL`이면 serving으로 넘어가지 않는다.

## 8. Serving 실행

```bash
make up
make gate-load
make gate-context
make smoke
make status
```

context checkpoint는 `262144 → 524288 → 786432 → 1048576` 순서다. `make baseline`으로 다음 DSpark lane의 target-only fixture를 남긴다.
이 lane에는 DSpark gate를 적용하지 않는다. `make calibrate`로 현재 `VLLM_MAX_NUM_SEQS` 후보를 검증한다.

`make up`은 head profile을 먼저 시작하고 SSH로 worker1–3 profile을 병렬 시작한다.
`K3-HEALTH PASS ... expected_ranks=32`와 `K3-UP-PASS`가 모두 보여야 한다.

## 9. PASS 기준

- load: 원본 MXFP4 checkpoint profile/first token과 32 rank 초기화 PASS
- context: `262144 → 524288 → 786432 → 1048576` 각 요청이 1 token을 생성
- memory: context/calibration 종료 후 모든 GPU free HBM ≥5%
- topology: lane 01은 PP4, lane 02–05는 DP4/EP32 stamp가 현재 env와 일치
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
잔존 process가 없을 때만 `05`로 이동한다. 앞 lane FAIL을 뒤 lane 실행으로
덮지 않으며, 실행하지 않은 lane은 FAIL이 아니라 `INCOMPLETE`로 둔다.
