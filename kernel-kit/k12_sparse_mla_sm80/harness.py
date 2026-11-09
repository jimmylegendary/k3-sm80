#!/usr/bin/env python3
"""k12_sparse_mla_sm80 — K3.1 DSA sparse MLA (SM80 Triton split-KV) 단독 검증.

대상 (vendor/dsa_ops = **이미지에 실리는 패치본**: vLLM a3c14efb + docker/patches-k31/triton-mla-sparse-sm80.patch
      (sha256 74675e99…7ed8), 원천 PR #47629 head 064801dd · 원저자 haosdent/thomaslwang — PR 원본이 아니다):
  triton_mla_sparse_kernel.triton_mla_sparse_attention(q, kv, indices, sm_scale, num_kv_splits, sm_count, mid_out_getter)
    q       (T, heads_local, 576) bf16   = absorbed latent query (K3.1: heads_local = 96/TP8 = 12)
    kv      (S, 1, 576) bf16             = 물리 latent 행 풀 (V = 앞 512 lane 재사용)
    indices (T, 1, 2048) int32           = indexer top-k 의 물리 행 (<0 또는 ≥S = 무효)
    →       (T, heads_local, 512) bf16
  백엔드(`triton_mla_sparse.py`)가 부르는 형태 그대로 — `sm_count` 는 백엔드가 캐시한 SM 수를 넘기고(A100=108), split 수는 커널의
  `_choose_num_kv_splits` 가 정한다(decode b1 → 16 splits, b8 → 8, prefill t256 → 1 = single-pass `_sparse_mla_kernel_final`).
  패치본이 PR 원본에 더한 두 가지를 하네스가 그대로 쓴다:
    · int64 승격 — `indices.to(tl.int64)` / `cur_q = program_id(0).to(tl.int64)`. packed hybrid KV row 에서는 물리 row id 가 토큰
      위치의 ~26.75배라 `row*576` 이 ~139k 토큰부터(row id ≥ 3,728,270 = 2^31/576) int32 를 넘는다 → PR 원본은 wrap 해서 잘못된
      주소를 읽는다. REQUIRED `dec_b1_ctx160k_packed` 가 블록 stride 를 벌린 풀(`block_stride_rows`)로 이 구간을 직접 읽는다.
    · split-KV 스크래치 workspace 주입 — 백엔드는 공용 workspace 에서 `mid_out_getter` 로 스크래치를 받고, 예약량은
      `max_split_mid_out_numel(heads, topk, sm_count)` 이다(잠근 뒤 증설 불가). 하네스는 auto(프로덕션 디스패치) 변종에 같은 크기의
      예약 스크래치(ReservedScratch — 한 덩어리 예약, 매 호출 NaN 으로 오염 = '더러운 workspace')를 주입해 ① 예약식이 실제 요청을
      덮는지 ② 커널이 스크래치를 전부 쓴 뒤에만 읽는지 본다. 강제 split 변종은 예약 밖(일반 할당)으로 돈다.

참조 = common/dsa_ref.sparse_mla (fp32): 선택 행 gather → q·kᵀ·192^-0.5 → softmax → ·V(앞 512 lane), 무효 행 마스크,
  유효 행이 없는 질의는 출력 0. 입력은 커널과 오라클이 같은 bf16 텐서를 공유한다.

판정 (docs/PLAN.md §S2 4게이트):
  NUMERIC   커널 출력(bf16) vs fp32 오라클: tolerance.TOL[k12] (rmse_rel ≤ 5e-3, max_abs ≤ 5e-2) +
            NaN/Inf 즉시 FAIL. 케이스마다 split 변종을 전부 대조한다 — decode: auto/1/4 splits, prefill: auto/2 splits
            (auto 는 decode 에서 split+merge 커널, prefill 에서 single-pass 커널을 타므로 변종이 반대 경로를 덮는다).
            짧은 프리필(pre_short_t64)은 top-k 2048 중 대부분이 -1 sentinel 이고 마지막 질의는 전부 -1(패딩 행 → 출력 정확히 0).
            kv 풀의 미사용 블록·마지막 블록 tail 은 독극물(1e4)로 채워 인덱스/마스크 실수가 출력으로 드러나게 한다.
  SHAPES    REQUIRED_SHAPES 11케이스 (decode b1/8 × ctx 8k/128k/256k + packed-row 1 + prefill t256 × ctx 8k/128k/256k + short)
            dec_b1_ctx160k_packed 는 물리 row id 가 int32 오프셋 한계(3,728,270)를 넘는 구간을 실제로 읽는다 — 선택 행 중 한계 너머가
            0 개이거나 최대 오프셋이 2^31 미만이면 케이스가 공회전(vacuous)이므로 `…/int32_zone` entry 가 FAIL 한다.
            GPU 메모리 추정 5.66 GB(풀 5.05 GB + 작업 버퍼; evidence.packed_pool, 목표 ≤ 8 GiB).
  SANITIZER compute-sanitizer memcheck — 소형 child 2종(decode b2·ctx1000·topk256 / short prefill t16)
  PERF      **측정 전용**(PERF_POLICY k12 measure_only, cv_max 0.15): 대표 케이스 dec_b1_ctx256k 의 CV 만 판정,
            전 케이스 median·gather 루프라인 효율은 evidence. 속도 floor 는 A100 실측 후 사전등록.

모드: (기본) GPU parent/child / --cpu-dry-run 순수 python 자체점검(INCOMPLETE 상한) /
      --interpret CPU Triton 인터프리터 소형 대조 → evidence.interpret_cpu (**게이트 판정 아님**).
      인터프리터는 packed-row 케이스를 **실제 기하(row id ≥ 3.7M, 풀 4.3 GB)** 로 돈다 — 풀은 가상 메모리로만 잡고 사용 블록 행만
      채우므로 RSS 는 수백 MB 다(`int_decode_b1_ctx330_packed`).
exit: 0 FAIL 없음 / 1 FAIL / 2 GPU 사전조건 미충족 / 3 내부 오류
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

from common import dsa_gate, dsa_ref, env_check, report, shapes, timing, tolerance  # noqa: E402

KID = "k12_sparse_mla_sm80"
PINS = {
    "vllm_pin": "a3c14efb",
    "dsa_ops": "a3c14efb+triton-mla-sparse-sm80.patch(sha256 74675e99…7ed8; origin PR#47629@064801dd haosdent/thomaslwang)",
    "hf_k31": "moonshotai/Kimi-K3.1-Pro@d9e99b07",
}
KERNEL_MODULE = ("vendor/dsa_ops = 배포 패치본(vLLM a3c14efb + docker/patches-k31/triton-mla-sparse-sm80.patch "
                 "sha256 74675e99…7ed8; 원천 PR #47629 head 064801dd)")
KERNEL_FILES = ("triton_mla_sparse_kernel.py",)
CHILD_SCHEMA = "k12-child/v1"
CASE_ORDER = tuple(shapes.SPARSE_MLA_REQUIRED_CASES)
PERF_CASE = "dec_b1_ctx256k"

DQ, DV = shapes.MLA_CACHE_D, shapes.MLA_KV_LORA                    # 576, 512
HEADS_LOCAL, TOPK, BLOCK = shapes.DSA_HEADS_LOCAL, shapes.DSA_TOPK, shapes.DSA_BLOCK_SIZE
SCALE = (shapes.MLA_QK_NOPE + shapes.MLA_QK_ROPE) ** -0.5          # 192^-0.5 (k04 와 동일: 모델 MLA scale)
POISON = 1.0e4        # 미사용 블록/tail/블록 사이 패딩 채움 (bf16 표현 가능) — 잘못 읽히면 softmax 를 지배한다
A100_SM_COUNT = 108   # 백엔드가 넘기는 sm_count (interpret 모드에서도 같은 값으로 split 휴리스틱을 재현)
INT32_ROW_LIMIT = shapes.DSA_ROW_INT32_LIMIT   # 3,728,270 = 2^31 // 576 : 이 row id 부터 `row*576` 이 int32 를 넘는다
GPU_BUDGET_BYTES = 8 * 2**30                   # REQUIRED 케이스 GPU 메모리 목표 상한 (A100 80 GB 중 대형 풀 1개 + 작업 버퍼)
RANDN_CHUNK_ELEMS = 1 << 26                    # dsa_gate.gpu_randn_bf16 청크(fp32 난수 + bf16 변환) 크기

SANITIZER_CASES = {
    "san_decode_b2_ctx1000": dict(mode="decode", b=2, ctx=1000, topk=256),
    "san_prefill_short_t16": dict(mode="prefill", t=16, ctx=16, short=True, topk=256),
}
INTERPRET_CASES = {
    # topk 256: auto 가 2-way split 을 골라 예약 workspace(ReservedScratch)로 스크래치를 받는 경로까지 돈다 (topk 64 면 auto=single-pass)
    "int_decode_b2_ctx512": dict(mode="decode", b=2, ctx=512, heads=4, topk=256),
    "int_prefill_t8_ctx512": dict(mode="prefill", t=8, ctx=512, heads=4, topk=64),
    "int_prefill_short_t8": dict(mode="prefill", t=8, ctx=8, heads=4, topk=64, short=True),
    # packed hybrid KV row 의 실제 기하(stride 576, 블록 stride 1712 행, row id ≥ 3.7M)를 CPU 에서 재현: 풀(≈4.3 GB)은 가상 메모리로만
    # 잡고 사용 블록 6개(≈0.4 MB)만 채운다. PR 원본·int64 승격을 뺀 변종은 여기서 SIGSEGV 하거나 오답이다(tests/kernel/test_dsa_ops_vendor.py).
    "int_decode_b1_ctx330_packed": dict(mode="decode", b=1, ctx=330, heads=4, topk=256,
                                        block_stride_rows=shapes.DSA_PACKED_BLOCK_STRIDE_ROWS,
                                        pool_blocks=2200, blocks_at_top=True),
}


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def variants_for(cfg: dict) -> list:
    """검증할 num_kv_splits 변종: None(auto, 프로덕션) + 반대 경로를 덮는 명시값."""
    return [None, 1, 4] if cfg["mode"] == "decode" else [None, 2]


def query_lengths(cfg: dict) -> list[int]:
    """질의별 후보 위치 수 (인과: 위치 p 의 질의는 [0, p] 를 본다)."""
    if cfg["mode"] == "decode":
        return [max(2, cfg["ctx"] - 1 - ((i * 37) % 61)) for i in range(cfg["b"])]
    t, ctx = cfg["t"], cfg["ctx"]
    return [ctx - t + j + 1 for j in range(t)]


class ScratchReservationError(RuntimeError):
    """예약한 스크래치보다 큰 요청 — 잠긴 vLLM workspace 에서 증설이 불가능한 상황과 같다."""


class ReservedScratch:
    """백엔드의 split-KV 스크래치 workspace 모사 (`mid_out_getter` 로 커널에 주입).

    vLLM 백엔드는 `max_split_mid_out_numel(heads, topk, sm_count)` 만큼 fp32 를 한 번 예약하고(workspace 를 잠근 뒤엔 증설 불가)
    매 호출 그 앞부분의 view 를 받는다. 이 클래스는 같은 크기를 한 덩어리로 예약하고, 요청이 예약을 넘으면
    ScratchReservationError 로 실패한다(예약식이 실제 요청을 못 덮으면 첫 decode 에서 죽는 것과 동치).
    poison=True 면 매 호출 view 를 NaN 으로 채워 '다른 용도가 남긴 더러운 workspace' 를 흉내 낸다 — 커널이 스크래치를
    전부 쓰기 전에 읽으면 출력이 NaN 이 되어 NUMERIC 에서 드러난다(성능 측정용 인스턴스는 poison=False).
    """

    def __init__(self, torch_mod, reserved_numel: int, device: str, poison: bool = True):
        self.reserved = int(reserved_numel)
        self.buf = torch_mod.empty(self.reserved, dtype=torch_mod.float32, device=device)
        self.poison = poison
        self.requests: list[int] = []

    def __call__(self, shape):
        numel = 1
        for dim in shape:
            numel *= int(dim)
        self.requests.append(numel)
        if numel > self.reserved:
            raise ScratchReservationError(
                f"스크래치 예약 부족: 요청 {numel} > 예약 {self.reserved} fp32 원소 (shape={tuple(shape)}) — "
                "max_split_mid_out_numel 가 실제 요청을 덮지 못한다")
        view = self.buf[:numel].view(*[int(d) for d in shape])
        if self.poison:
            view.fill_(float("nan"))
        return view


def packed_geometry(cfg: dict) -> dict:
    """풀 기하와 GPU 메모리 추정 (순수 python — dry-run·단위 시험·evidence 공용).

    block_stride_rows 가 없으면 dense 풀(stride = BLOCK). 풀은 bf16 `(pool_blocks*stride, 1, 576)` 한 개이고, 사용 블록의 행만
    난수로 채워 index_copy 한다 → 작업 버퍼 = 사용 행 난수(`used_rows*576*2 B`) + 청크(fp32 난수 + bf16 변환) + 소량.
    """
    stride = cfg.get("block_stride_rows", BLOCK)
    lens = query_lengths(cfg)
    seq_len = lens if cfg["mode"] == "decode" else [cfg["ctx"]]
    used_blocks = sum(_cdiv(n, BLOCK) for n in seq_len)
    pool_blocks = cfg.get("pool_blocks", used_blocks + 2)
    pool_rows = pool_blocks * stride
    pool_bytes = pool_rows * DQ * 2
    used_bytes = used_blocks * BLOCK * DQ * 2 if stride != BLOCK else 0
    chunk_bytes = RANDN_CHUNK_ELEMS * (4 + 2)
    max_row = (pool_blocks - 1) * stride + BLOCK - 1
    first_zone_block = _cdiv(INT32_ROW_LIMIT, stride)            # 첫 행이 한계 이상인 가장 작은 블록 id
    return {"block_stride_rows": stride, "used_blocks": used_blocks, "pool_blocks": pool_blocks, "pool_rows": pool_rows,
            "pool_bytes": pool_bytes, "pool_gib": round(pool_bytes / 2**30, 3),
            "est_gpu_bytes": pool_bytes + used_bytes + chunk_bytes + (16 << 20),
            "est_gpu_gib": round((pool_bytes + used_bytes + chunk_bytes + (16 << 20)) / 2**30, 3),
            "max_row": max_row, "max_offset_elems": max_row * DQ + DQ - 1,
            "int32_row_limit": INT32_ROW_LIMIT, "first_zone_block": first_zone_block,
            "blocks_past_limit": max(0, pool_blocks - first_zone_block),
            "crosses_int32": max_row * DQ + DQ - 1 >= 2**31}


def run_case(cfg: dict, case: str, device: str, do_timing: bool, warmup: int, reps: int,
             kernel_only: bool = False) -> dict:
    """케이스 하나. kernel_only=True 는 sanitizer child 용 — 오라클·대조 없이 모든 split 변종 커널만 실행한다."""
    import torch
    from vendor.dsa_ops.triton_mla_sparse_kernel import (
        _choose_num_kv_splits, max_split_mid_out_numel, triton_mla_sparse_attention)
    seed = shapes.stable_seed(KID, case)
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    on_cuda = device.startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()               # 실측 피크(evidence.gpu_mem) — packed_geometry 추정치와 대조
    heads, topk = cfg.get("heads", HEADS_LOCAL), cfg.get("topk", TOPK)
    decode = cfg["mode"] == "decode"
    lens = query_lengths(cfg)                              # 질의별 후보 수 (decode: 시퀀스 길이)
    ntok = len(lens)
    seq_len = lens if decode else [cfg["ctx"]]              # 시퀀스별 컨텍스트 길이
    nblk = [_cdiv(length, BLOCK) for length in seq_len]
    n_used = sum(nblk)
    stride_rows = cfg.get("block_stride_rows", BLOCK)       # 블록 시작 row 간격 (BLOCK = dense 풀, 그 이상 = packed hybrid row 흉내)
    packed = stride_rows != BLOCK
    pool_blocks = cfg.get("pool_blocks", n_used + 2)        # 사용 블록 + 미사용 2 (기본)
    if cfg.get("blocks_at_top"):                            # 사용 블록을 풀의 맨 위 id 로 몰아 둔다 (소형 인터프리터 케이스)
        perm = (pool_blocks - (n_used + 2)) + torch.randperm(n_used + 2, generator=cpu_gen)
    else:
        assert pool_blocks == n_used + 2, "pool_blocks 는 blocks_at_top 과 함께만 지정한다"
        perm = torch.randperm(pool_blocks, generator=cpu_gen)
    bt = torch.zeros((len(seq_len), max(nblk)), dtype=torch.int64)
    o = 0
    for i, n_i in enumerate(nblk):
        bt[i, :n_i] = perm[o:o + n_i]
        o += n_i
    t_build = time.monotonic()
    q = dsa_gate.gpu_randn_bf16((ntok, heads, DQ), gen, device)
    pool_rows = pool_blocks * stride_rows
    if packed:
        # 블록 사이 패딩 row 는 다른 그룹(index-K·KDA state)의 페이지 자리 — GPU 는 독극물로 채워 wrap 된 주소 읽기를 드러내고, CPU 는 가상
        # 메모리(lazily zero)를 건드리지 않으려 채우지 않는다(사용 블록 행만 쓴다). 풀이 2^31 원소를 넘으므로 고급 인덱싱 대신
        # 블록 단위 기본 인덱싱 view 에 copy_/fill_ 한다(64비트 오프셋이 보장되는 가장 단순한 경로).
        pool = torch.empty((pool_rows, 1, DQ), dtype=torch.bfloat16, device=device)
        if on_cuda:
            pool.fill_(POISON)
        blocks = pool.view(pool_blocks, stride_rows, DQ)
        data = dsa_gate.gpu_randn_bf16((n_used * BLOCK, DQ), gen, device).view(n_used, BLOCK, DQ)
        for i, blk in enumerate(perm[:n_used].tolist()):
            blocks[blk, :BLOCK].copy_(data[i])
        del data
        # 독극물: 미사용 블록 + 각 시퀀스 마지막 블록의 tail
        for blk in perm[o:o + 2].tolist():
            blocks[blk, :BLOCK].fill_(POISON)
        for i, length in enumerate(seq_len):
            tail = nblk[i] * BLOCK - length
            if tail:
                blocks[int(bt[i, nblk[i] - 1]), BLOCK - tail:BLOCK].fill_(POISON)
    else:
        pool = dsa_gate.gpu_randn_bf16((pool_rows, 1, DQ), gen, device)
        # 독극물: 미사용 블록 + 각 시퀀스 마지막 블록의 tail
        poison_rows = [perm[o:o + 2].mul(BLOCK).unsqueeze(1) + torch.arange(BLOCK).unsqueeze(0)]
        for i, length in enumerate(seq_len):
            tail = nblk[i] * BLOCK - length
            if tail:
                last = int(bt[i, nblk[i] - 1])
                poison_rows.append(torch.arange(last * BLOCK + BLOCK - tail, last * BLOCK + BLOCK).unsqueeze(0))
        pool[torch.cat([r.reshape(-1) for r in poison_rows]).to(device), 0, :] = POISON
    # 인덱스: 질의마다 후보 위치에서 min(topk, 후보 수) 개를 비복원 추출 → 물리 행으로 변환, 나머지 -1
    logical = torch.full((ntok, topk), -1, dtype=torch.int64)
    for j, count in enumerate(lens):
        take = min(topk, count)
        logical[j, :take] = torch.randperm(count, generator=cpu_gen)[:take]
        if cfg.get("short"):
            logical[j] = logical[j][torch.randperm(topk, generator=cpu_gen)]      # -1 이 행 중간에도 섞이게
    if cfg.get("short"):
        logical[-1, :] = -1                                                       # 패딩 행: 유효 인덱스 0개
    seq_of = torch.arange(ntok) if decode else torch.zeros(ntok, dtype=torch.int64)
    valid = logical >= 0
    pos = logical.clamp(min=0)
    phys = bt[seq_of[:, None], pos // BLOCK] * stride_rows + pos % BLOCK
    indices = torch.where(valid, phys, torch.full_like(phys, -1)).to(torch.int32).to(device).view(ntok, 1, topk)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count if on_cuda else A100_SM_COUNT
    auto_splits = _choose_num_kv_splits(ntok, -(-heads // min(16, heads)), topk, sm_count)
    reserved = max_split_mid_out_numel(heads, topk, sm_count)               # 백엔드가 예약하는 스크래치 상한(fp32 원소)
    ws = ReservedScratch(torch, reserved, device)
    # int32 한계 구간 점검: 선택된 물리 row 중 `row*576 >= 2^31` 인 것이 있어야 packed 케이스가 공회전하지 않는다
    selected = phys[valid]
    zone = {"selected_rows": int(selected.numel()),
            "max_selected_row": int(selected.max()) if selected.numel() else -1,
            "selected_rows_past_int32_limit": int((selected >= INT32_ROW_LIMIT).sum()) if selected.numel() else 0,
            "int32_row_limit": INT32_ROW_LIMIT}
    zone["max_selected_offset_elems"] = zone["max_selected_row"] * DQ + DQ - 1 if selected.numel() else -1
    require_zone = cfg.get("require_past_int32", packed)

    if kernel_only:
        for nks in variants_for(cfg):
            triton_mla_sparse_attention(q, pool, indices, sm_scale=SCALE, num_kv_splits=nks, sm_count=sm_count,
                                        mid_out_getter=ws if nks is None else None)
        if on_cuda:
            torch.cuda.synchronize()
        return {"schema_version": CHILD_SCHEMA, "case": case, "shape": dict(cfg), "status": "PASS",
                "entries": [], "timing": None,
                "extra": {"seed": seed, "kernel_only": True, "auto_num_kv_splits": auto_splits}}
    ref, _ = dsa_ref.sparse_mla(q, pool, indices, SCALE, DV)
    build_s = time.monotonic() - t_build
    entries, variants_ev = [], {}
    if require_zone:
        viol = []
        if zone["selected_rows_past_int32_limit"] == 0:
            viol.append(f"선택된 물리 row 중 int32 한계({INT32_ROW_LIMIT}) 이상이 0개 — packed 케이스가 공회전(vacuous)")
        if zone["max_selected_offset_elems"] < 2**31:
            viol.append(f"최대 선택 오프셋 {zone['max_selected_offset_elems']} < 2^31 — int32 wrap 구간을 읽지 않음")
        entries.append({"case": f"{case}/int32_zone", "metrics": dict(zone), "violations": viol, "passed": not viol})
    for nks in variants_for(cfg):
        label = "auto" if nks is None else f"splits={nks}"
        # auto = 프로덕션 디스패치 → 예약된 workspace 에서 스크래치를 받는다. 강제 변종은 예약식이 덮지 않는 모양이라 일반 할당.
        try:
            out = triton_mla_sparse_attention(q, pool, indices, sm_scale=SCALE, num_kv_splits=nks, sm_count=sm_count,
                                              mid_out_getter=ws if nks is None else None)
        except ScratchReservationError as exc:
            entries.append({"case": f"{case}/out[{label}]", "metrics": {"reserved_numel": reserved,
                            "requests": list(ws.requests)}, "violations": [str(exc)], "passed": False})
            variants_ev[label] = {"error": "ScratchReservationError"}
            continue
        if on_cuda:
            torch.cuda.synchronize()
        res = tolerance.compare(KID, ref, out.float(), case=f"{case}/out[{label}]")
        if cfg.get("short"):                                                       # 패딩 행은 정확히 0
            zero_ok = bool((out[-1].float() == 0).all())
            if not zero_ok:
                res.violations.append("전부 -1 인 패딩 행의 출력이 0 이 아님")
                res.passed = False
            res.metrics["padding_row_exact_zero"] = zero_ok
        entries.append({"case": res.metrics["case"], "metrics": res.metrics, "violations": res.violations,
                        "passed": res.passed})
        variants_ev[label] = {"rmse_rel": res.metrics.get("rmse_rel"), "max_abs": res.metrics.get("max_abs")}
        del out
    stats = None
    if do_timing and on_cuda:
        ws_timed = ReservedScratch(torch, reserved, device, poison=False)       # 측정용: 오염 fill 이 타이밍에 끼지 않게
        stats = timing.measure(lambda: triton_mla_sparse_attention(
            q, pool, indices, sm_scale=SCALE, num_kv_splits=None, sm_count=sm_count, mid_out_getter=ws_timed),
            warmup=warmup, reps=reps)
    geo = packed_geometry(cfg)
    extra = {"seed": seed, "heads": heads, "topk": topk, "tokens": ntok, "pool_rows": int(pool.shape[0]),
             "auto_num_kv_splits": auto_splits, "sm_count": sm_count, "variants": variants_ev,
             "valid_per_query_head": [int(v) for v in valid.sum(1)[:4]], "build_s": round(build_s, 1),
             "workspace": {"reserved_numel": reserved, "reserved_mib": round(reserved * 4 / 2**20, 3),
                           "auto_requests": list(ws.requests),
                           "bound_covers_requests": all(r <= reserved for r in ws.requests)},
             "pool": {k: geo[k] for k in ("block_stride_rows", "pool_blocks", "pool_gib", "est_gpu_gib",
                                          "max_row", "blocks_past_limit")},
             "int32_zone": zone}
    if on_cuda:
        extra["gpu_mem"] = {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                            "estimated_gib": geo["est_gpu_gib"], "budget_gib": GPU_BUDGET_BYTES / 2**30}
    del ref, pool, q
    if on_cuda:
        torch.cuda.empty_cache()
    return {"schema_version": CHILD_SCHEMA, "case": case, "shape": dict(cfg),
            "status": "PASS" if all(e["passed"] for e in entries) else "FAIL",
            "entries": entries, "timing": stats, "extra": extra}


def _run_child(case: str, child_out: Path, warmup: int, reps: int) -> int:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 0):
        raise RuntimeError(f"k12 requires SM80, got {capability}")
    cfg = SANITIZER_CASES.get(case) or shapes.SPARSE_MLA_REQUIRED_CASES.get(case)
    if cfg is None:
        raise ValueError(f"unknown case: {case}")
    print(f"K12-CHILD start case={case} cfg={cfg}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    san = case.startswith("san_")           # sanitizer child: 커널만(오라클·타이밍 생략)
    payload = run_case(cfg, case, "cuda", do_timing=not san, warmup=warmup, reps=reps, kernel_only=san)
    child_out.parent.mkdir(parents=True, exist_ok=True)
    child_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"K12-CHILD done case={case} status={payload['status']}", flush=True)
    return 0 if payload["status"] == "PASS" else 1


# ===========================================================================
# --cpu-dry-run : 순수 python 자체점검
# ===========================================================================
def dry_selfcheck() -> dict:
    import random
    rnd = random.Random(shapes.stable_seed(KID, "dry"))
    t, h, dq, s_kv, topk, dv = 3, 2, 12, 20, 8, 8
    q = [[[rnd.gauss(0, 1) for _ in range(dq)] for _ in range(h)] for _ in range(t)]
    kv = [[rnd.gauss(0, 1) for _ in range(dq)] for _ in range(s_kv)]
    idx = [rnd.sample(range(s_kv), topk) for _ in range(t)]
    idx[1][5:] = [-1, -1, -1]
    idx[2] = [-1] * topk
    scale = dq ** -0.5
    out = dsa_ref.sparse_mla_py(q, kv, idx, scale, dv)
    checks: dict[str, bool] = {}
    perm_idx = [row[::-1] for row in idx]                       # 인덱스 순서 불변성
    out2 = dsa_ref.sparse_mla_py(q, kv, perm_idx, scale, dv)
    flat = lambda o: [v for tt in o for hh in tt for v in hh]   # noqa: E731
    checks["index_order_invariant"] = all(abs(a - b) < 1e-12 for a, b in zip(flat(out), flat(out2)))
    checks["empty_query_is_zero"] = all(v == 0.0 for hh in out[2] for v in hh)
    checks["softmax_convex_hull"] = all(min(kv[i][d] for i in idx[0]) - 1e-9 <= out[0][0][d]
                                        <= max(kv[i][d] for i in idx[0]) + 1e-9 for d in range(dv))
    ref = flat(out)
    checks["tol_pass_case"] = tolerance.compare(KID, ref, [v * (1 + 1e-3) for v in ref], "dry_ok").passed
    checks["tol_fail_case"] = not tolerance.compare(KID, ref, [v * 1.05 for v in ref], "dry_bad").passed
    checks["required_shapes_cover_modes"] = {c["mode"] for c in shapes.SPARSE_MLA_REQUIRED_CASES.values()} == {"decode", "prefill"}
    # packed-row 케이스(int32 한계 너머 row id)의 기하·GPU 메모리를 순수 python 으로 점검 — 실제 읽기는 GPU/인터프리터 몫
    geo = packed_geometries()
    checks["int32_row_limit_is_2p31_over_576"] = INT32_ROW_LIMIT == 2**31 // DQ == 3_728_270
    checks["packed_case_registered"] = bool(geo)
    checks["packed_case_pool_crosses_int32_offset"] = bool(geo) and all(g["crosses_int32"] and g["blocks_past_limit"] > 0
                                                                       for g in geo.values())
    checks["packed_case_gpu_memory_within_budget"] = bool(geo) and all(g["est_gpu_bytes"] <= GPU_BUDGET_BYTES
                                                                       for g in geo.values())
    return {"mode": "cpu-dry-run(순수 python, torch/GPU 불필요)", "checks": checks,
            "selfcheck": "PASS" if all(checks.values()) else "FAIL"}


def packed_geometries() -> dict:
    """REQUIRED 중 block_stride_rows 로 벌린 풀을 쓰는 케이스들의 기하·메모리 추정 (evidence 로도 기록)."""
    return {name: packed_geometry(cfg) for name, cfg in shapes.SPARSE_MLA_REQUIRED_CASES.items()
            if cfg.get("block_stride_rows", BLOCK) != BLOCK}


def kernel_evidence() -> dict:
    """리포트 evidence['kernel'] — 검증 대상이 PR 원본이 아니라 배포 패치본임을 파일 해시·int64 마커로 남긴다."""
    return {"module": KERNEL_MODULE, "entry_point": "triton_mla_sparse_attention",
            "call_contract": "mid_out_getter(auto 변종, 예약 = max_split_mid_out_numel) + int64 승격 커널",
            "files": dsa_gate.kernel_provenance(KERNEL_FILES)}


def build_dry_report() -> dict:
    result = report.KernelReport(KID, dry_run=True, pins=PINS)
    sc = dry_selfcheck()
    result.add_evidence("dry_run", sc)
    result.add_evidence("kernel", kernel_evidence())
    result.add_evidence("constants", {"heads_local": HEADS_LOCAL, "topk": TOPK, "dq": DQ, "dv": DV,
                                      "sm_scale": SCALE, "block": BLOCK, "sm_count": A100_SM_COUNT,
                                      "int32_row_limit": INT32_ROW_LIMIT})
    result.add_evidence("packed_pool", {"note": "REQUIRED packed-row 케이스의 풀 기하·GPU 메모리 추정(목표 ≤ 8 GiB)",
                                        "gpu_budget_gib": GPU_BUDGET_BYTES / 2**30, "cases": packed_geometries()})
    result.add_evidence("required_shapes", shapes.REQUIRED_SHAPES[KID])
    result.set_gate("NUMERIC", "SKIP", f"cpu-dry-run: 실커널 미실행 — 순수 python 참조·판정 자체점검 "
                    f"{len(sc['checks'])}건 {sc['selfcheck']}")
    result.set_gate("SANITIZER", "SKIP", "cpu-dry-run: compute-sanitizer 는 A100 실행 필요")
    result.set_gate("SHAPES", "SKIP", f"cpu-dry-run: {len(CASE_ORDER)}케이스 등록만 확인, 미실행")
    result.set_gate("PERF", "SKIP", "cpu-dry-run: CUDA 타이밍 미측정 (측정 전용 정책)")
    return result.as_dict()


# ===========================================================================
# --interpret : CPU Triton 인터프리터 대조 (게이트 판정 아님)
# ===========================================================================
def run_interpret(report_dir: Path) -> int:
    os.environ["TRITON_INTERPRET"] = "1"
    from common import triton_interp
    info = triton_interp.install()
    results, ok = [], True
    for name, cfg in INTERPRET_CASES.items():
        t0 = time.monotonic()
        try:
            payload = run_case(cfg, name, "cpu", do_timing=False, warmup=0, reps=0)
            payload["duration_s"] = round(time.monotonic() - t0, 1)
        except Exception as exc:
            traceback.print_exc()
            payload = {"case": name, "shape": dict(cfg), "status": "ERROR",
                       "error": f"{type(exc).__name__}: {exc}"[:500], "entries": []}
        ok &= payload["status"] == "PASS"
        results.append(payload)
        print(f"K12-INTERPRET case={name} status={payload['status']}", flush=True)
    result = report.KernelReport(KID, dry_run=True, pins=PINS)
    result.add_evidence("interpret_cpu", {
        "note": "CPU 인터프리터 대조 — PASS 근거 아님(정직한 상한). GPU 게이트는 전부 SKIP 유지.",
        "overall": "PASS" if ok else "FAIL", "interpreter": info,
        "cases": [{k: p.get(k) for k in ("case", "shape", "status", "error", "duration_s", "extra")}
                  | {"entries": [{"case": e["case"], "passed": e["passed"], "violations": e["violations"],
                                  "metrics": e["metrics"]} for e in p["entries"]]} for p in results]})
    for gate in report.GATES:
        result.set_gate(gate, "SKIP", f"--interpret: CPU 인터프리터 대조 {('PASS' if ok else 'FAIL')} — "
                        "게이트 판정 아님(GPU 실행 필요)")
    dsa_gate.write_payload(KID, result.as_dict(), report_dir)
    print(f"K12-INTERPRET-OVERALL {'PASS' if ok else 'FAIL'} (게이트 미반영) report={report_dir / (KID + '.json')}")
    return 0 if ok else 1


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--interpret", action="store_true",
                    help="TRITON_INTERPRET=1 CPU 대조(게이트 판정 아님; 기존 리포트와 같은 이름으로 기록)")
    ap.add_argument("--report-dir", type=Path, default=KIT / "kernel-report")
    ap.add_argument("--phase0-verdict", default=None)
    ap.add_argument("--twin", default=None)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--cases", nargs="*", default=None, help="REQUIRED 케이스 부분 실행(SHAPES 는 FAIL 이 된다)")
    ap.add_argument("--child-case")
    ap.add_argument("--child-out", type=Path)
    args = ap.parse_args()

    if args.child_case:
        if args.child_out is None:
            ap.error("--child-case requires --child-out")
        return _run_child(args.child_case, args.child_out, args.warmup, args.reps)
    if args.interpret:
        return run_interpret(args.report_dir)
    if args.cpu_dry_run:
        payload = build_dry_report()
        dsa_gate.write_payload(KID, payload, args.report_dir)
        sc = payload["evidence"]["dry_run"]["selfcheck"]
        print(f"K12-OVERALL INCOMPLETE mode=cpu-dry-run selfcheck={sc} report={args.report_dir / (KID + '.json')}")
        return report.EXIT_OK if sc == "PASS" else report.EXIT_ERR
    extra = {"kernel": kernel_evidence(),
             "call_contract": {"sm_scale": SCALE, "heads_local": HEADS_LOCAL, "topk": TOPK, "dq": DQ, "dv": DV,
                               "kv": "(S,1,576) bf16 flat physical rows", "indices": "(T,1,topk) int32, -1 invalid",
                               "mid_out_getter": "auto 변종에 ReservedScratch(max_split_mid_out_numel) 주입, 매 호출 NaN 오염"},
             "packed_pool": {"gpu_budget_gib": GPU_BUDGET_BYTES / 2**30, "cases": packed_geometries()},
             "honest_limit": "A800/GLM 검증 커널을 K3.1 shape(heads 12·topk 2048)로 처음 실행 — 속도 미보증, 정확성만 판정"}
    return dsa_gate.run_parent(KID, Path(__file__), PINS, CASE_ORDER, PERF_CASE, list(SANITIZER_CASES), args, extra)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(report.EXIT_ERR)
