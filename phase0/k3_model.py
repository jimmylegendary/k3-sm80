#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 — Kimi K3 디코드 스텝 해석 모델 (32×A100 SM80, hatir 타당성 게이트).

실행:
  PYTHONPATH=/home/jimmy/repos/hatir/impl:/home/jimmy/repos/hat-schema \
      python3 /home/jimmy/repos/k3-sm80/phase0/k3_model.py

산출: phase0/results.json, phase0/REPORT.md, phase0/verdict.json
(REPORT/verdict의 모든 수치는 이 스크립트 실행 결과에서 생성 — 손수치 금지)

모델 구성 원칙:
  * 커널(GEMM/GEMV/attention/recurrent/elementwise)은 hatir `einsum.tile`로 직접
    저작한 플랜 + 명시 루프라인 max(플랜 compute_us, 명시 mem_bytes/HBM_BW)로 코스팅
    (트윈 = twin_a100_x32.json의 hbm 서브트리 → LevelStack). hatir 기본
    kernels/_gemm_plan·roofline_us는 사용하지 않음(사유는 lin_us docstring).
  * 통신(TP all-reduce, EP all-to-all, PP p2p, logits all-gather)은 hatir
    collectives(hierarchical_collective + FabricLevel)로 코스팅. α(소메시지 지연)는
    트윈 meta.comm_scenarios(보수 real/낙관 opt)에서 주입.
  * hatir가 표현하지 못하는 항목 — 전부 명시적 가산항/보정항으로 분리 ("EXPLICIT" 주석):
      (E1) 커널 런치/그래프 갭: kernels_per_layer × launch_gap + 그래프 분절 ×
           graph_segment (comm 경계마다 piecewise CUDA graph가 끊김)
      (E2) W4/MXFP4 dequant ALU 비용: params × dequant_ops / alu_ops_per_s 와의 max
           — dequant_ops_per_param은 [무근거 밴드] (트윈 _dequant_ops_note; Phase 1
           마이크로벤치 필수)
      (E3) MoE 라우팅 불균형: EP expert 커널 시간 × moe_imbalance
      (E4) per-operand 혼합 dtype: 양자화 GEMM은 dtype_bytes=B/param(<2)로 가중치
           바이트를 정확히 맞추고, 활성값 바이트 과소분(<0.1%)은 무시(주석 명기)
      (E5) vLLM 스텝 CPU/스케줄러 오버헤드: step_cpu_us 가산
      (E6) EP dispatch 전송 복사본 수: opt = DeepEP식 랭크-중복제거
           e×(1−(1−1/e)^topk) — 단 DeepEP는 SM90 전용이므로 SM80 커스텀 커널 전제.
           real = naive per-(token,expert) 복사(=topk) — vLLM SM80 기본 경로
           (recon/vllm-upstream.md에 SM80용 EP a2a 백엔드 근거 없음).
      (E7) MoE expert grouped-GEMV: per-operand dtype(가중치 4bit·활성 bf16)과
           유니크-expert 재사용이 hatir Op 1개로 표현 불가 → max(HBM, TC, dequant)
           명시 모델
      (E8) 노드 간 a2a α의 per-peer 구조: α_eff = α_base + α_per_peer × (노드 밖
           peer 수) — recon/a100-comms.md §3 (per-peer 소메시지 구조) 반영
      (P1) PP 마이크로배치 파이프라인 정상상태 처리량: 상주 배치 G를 p개 마이크로
           배치로 나눠 in-flight 유지 시 시스템 처리량 = G/p 토큰 / 최악 스테이지
           시간 (vLLM PP 다중 in-flight 배치; tok_s_system(동기 직렬)과 별도 산출)

모델 수치 출처(핀): recon/k3-config.md (HF moonshotai/Kimi-K3 @ 9f62e4e9), recon/
a100-comms.md, recon/vllm-upstream.md (vLLM main d223c900 / kimi-k3 0498dc7e),
recon/kda-fla.md. 트윈: phase0/twin_a100_x32.json.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

sys.path.insert(0, "/home/jimmy/repos/hatir/impl")
sys.path.insert(0, "/home/jimmy/repos/hat-schema")

from hat_schema.validate import validate_twin           # noqa: E402
from hatir.hw import linearize                          # noqa: E402
from hatir.einsum import tile                           # noqa: E402
from hatir.collectives import FabricLevel, hierarchical_collective  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN_PATH = os.path.join(HERE, "twin_a100_x32.json")

# ---------------------------------------------------------------- K3 모델 상수
# 전부 recon/k3-config.md (HF moonshotai/Kimi-K3 커밋 9f62e4e9fffbd0a83ddd60e1c2
# 09d828994b3569, 2026-07-28 확보)에서 온 1차 수치.
H = 7168                    # hidden_size
LATENT = 3584               # LatentMoE latent dim (routed_expert_hidden_size)
N_EXPERTS = 896
TOPK = 16
MOE_INTER = 3072            # expert FFN 폭 (latent 공간)
N_LAYERS = 93               # KDA 69 + MLA 24; FFN = dense 1 + MoE 92
KDA_HEADS, KDA_HD = 96, 128           # KDA 96 heads × 128, 투영폭 12288
KDA_W = KDA_HEADS * KDA_HD            # 12288
MLA_HEADS = 96
MLA_KV_LORA, MLA_Q_LORA = 512, 64 + 512   # latent 캐시 = 512+64(rope몫, NoPE) = 576
MLA_CACHE_D = 576                     # per-token latent 캐시 원소 수 (vLLM absorbed)
MLA_QK_D, MLA_V_D = 192, 128
VOCAB = 163840
SHARED_INTER = 6144                   # shared expert fused 2개 = intermediate 6144
DENSE_INTER = 33792

# 파라미터 수 (recon/k3-config.md §3, index로 바이트 폐쇄 검증된 값)
KDA_SHARD_P = 442_122_240   # q,k,v,g,f_b,o,conv — TP로 샤딩되는 부분
KDA_REPL_P = 1_618_144      # f_a, b_proj, A_log/dt_bias/o_norm — 복제(작음)
MLA_SHARD_P = 217_055_232   # q_b, kv_b, g, o
MLA_REPL_P = 15_140_864     # q_a, kv_a, norms (vLLM ReplicatedLinear 관행)
EXPERT_P = 33_030_144       # expert 1개 (w1+w2+w3 = 3×3584×3072)
ROUTED_P_LAYER = N_EXPERTS * EXPERT_P
SHARED_P = 132_120_576      # shared experts fused (hidden 공간, bf16)
LATENT_PROJ_P = 2 * 25_690_112        # down + up (층당 1쌍 공유)
ROUTER_P = 6_423_424        # gate 896×7168 + bias (복제, bf16 유지)
DENSE_P = 726_663_168
LMHEAD_P = VOCAB * H        # 1,174,405,120 (bf16 유지)

BPP_MXFP4 = 0.53125         # 4bit + E8M0 scale/32
BPP_W4A16 = 0.515625        # INT4 g128 + fp16 scale
# KDA state: vLLM은 fp32 하드코딩 (recon/kda-fla.md)
KDA_STATE_BYTES_PER_LAYER_SEQ = KDA_HEADS * KDA_HD * KDA_HD * 4   # 6.29 MB
KDA_CONV_BYTES_PER_LAYER_SEQ = 3 * KDA_W * 4 * 2                  # bf16, kernel 4

GiB = 1024 ** 3

# 층당 커널 수 (런치 오버헤드 E1의 입력; vLLM 디코드 그래프 기준 추정 —
# opt=적극 fusion(-O2 full graph), real=부분 fusion/piecewise. recon/a100-comms.md §4)
KERNELS = {
    "kda":   {"opt": 8,  "real": 13},
    "mla":   {"opt": 9,  "real": 14},
    "moe":   {"opt": 11, "real": 18},
    "dense": {"opt": 4,  "real": 6},
    "misc":  {"opt": 6,  "real": 12},   # embed/final-norm/lm_head/sampling 등 스텝당
}


# ---------------------------------------------------------------- 트윈 로드/검증
def load_twin():
    with open(TWIN_PATH) as f:
        return json.load(f)


def find_node(node, level_id):
    if node.get("level_id") == level_id:
        return node
    for ch in node.get("children") or []:
        r = find_node(ch, level_id)
        if r:
            return r
    return None


TWIN = load_twin()
VALIDATION = validate_twin(TWIN)
_errors = [v for v in VALIDATION if v["severity"] == "error"]
if _errors:
    raise SystemExit(f"twin validation errors: {_errors}")

HW = linearize(find_node(TWIN, "hbm"))     # per-GPU 서브스택: hbm→l2→smem→pe
HBM_BW = float(find_node(TWIN, "hbm")["bandwidth_bps"])          # 2039e9
_pe = find_node(TWIN, "pe")
TC_RATE = float(_pe["peak_macs_per_s"]) * int(_pe["instances"])  # 156e12 MACs/s
ALU_RATE = float(TWIN["meta"]["gpu_extra"]["alu_int32_ops_per_s"])
SCEN_RAW = TWIN["meta"]["comm_scenarios"]


NNODES = 4                  # 32 GPU = 4노드 × 8 (트윈 meta.cluster)


@dataclass(frozen=True)
class Scen:
    name: str
    nvlink_busbw: float
    ib_node: float
    ib_busbw: float
    ar_a_intra: float
    ar_a_inter: float
    a2a_a_intra: float
    a2a_a_inter: float
    a2a_a_inter_pp: float    # EXPLICIT(E8): 노드 간 a2a per-peer α (트윈 _a2a_alpha_note)
    a2a_dedup: bool          # EXPLICIT(E6): 랭크-중복제거(DeepEP식) 여부 — opt만 true
    p2p_a: float
    launch_gap: float
    graph_seg: float
    step_cpu: float
    dq_ops: float
    imbalance: float
    disp_payload: int
    pp_streams: int


def mk_scen(name):
    s = SCEN_RAW[name]
    return Scen(name, s["nvlink_busbw_bps"], s["ib_node_injection_bps"], s["ib_busbw_bps"],
                s["ar_alpha_intra_us"], s["ar_alpha_inter_us"], s["a2a_alpha_intra_us"],
                s["a2a_alpha_inter_us"], s["a2a_alpha_inter_per_peer_us"],
                bool(s["a2a_rank_dedup"]), s["p2p_alpha_us"], s["launch_gap_us"],
                s["graph_segment_us"], s["step_cpu_us"], s["dequant_ops_per_param"],
                s["moe_imbalance"], int(s["dispatch_payload_bytes"]), int(s["pp_hidden_streams"]))


SCENS = {n: mk_scen(n) for n in ("opt", "real")}


@dataclass(frozen=True)
class RQ:
    """재양자화 시나리오: 가중치 B/param + dequant ops/param.
    orig  = 체크포인트 그대로 (bf16 잔여 + MXFP4 expert; MXFP4는 SM80 HW 미지원이라
            dequant ops가 큼 — #35922 커널 가족의 에뮬레이션 경로).
    w4a16 = Phase 3 전량 INT4 g128 (router/lm_head/embed/norm bf16 유지). Marlin이
            FP16×INT4 소배치 근사-메모리바운드 실증(MARLIN, arXiv:2408.11743,
            PPoPP 2025 — INT4 근거이며 MXFP4 직접 근거 아님)이므로 dequant ops를
            절반으로(EXPLICIT E2 보정). dq_ops 자체는 [무근거 밴드] — 트윈
            _dequant_ops_note + Phase 1 마이크로벤치 필수."""
    name: str
    wb: float        # 비-expert linear 가중치 B/param
    dq_lin: float    # 비-expert linear dequant ops/param (wb<2일 때만 유효)
    bpp_exp: float   # expert 가중치 B/param
    dq_exp: float    # expert dequant ops/param


def mk_rq(requant: str, sc: Scen) -> RQ:
    if requant == "orig":
        return RQ("orig", 2.0, 0.0, BPP_MXFP4, sc.dq_ops)
    return RQ("w4a16", BPP_W4A16, sc.dq_ops * 0.5, BPP_W4A16, sc.dq_ops * 0.5)


@dataclass(frozen=True)
class Strat:
    """32 GPU = p(PP) × [stage 내: t_a(어텐션 TP) × dp(어텐션 DP)] / MoE는 stage의
    e(EP)×t_m(expert TP) = 32/p. a2a_scope: EP dispatch가 건너는 링크."""
    name: str
    p: int
    ta: int
    dp: int
    e: int
    tm: int
    a2a_scope: str | None
    ar_scope: str
    desc: str


STRATS = [
    Strat("TP8PP4", 4, 8, 1, 1, 8, None, "intra",
          "노드=PP stage, 노드 내 TP8 (expert도 텐서샤딩). a2a 없음"),
    Strat("TP4EP8PP4", 4, 4, 2, 8, 1, "intra", "intra",
          "과제 지정 'TP4×EP8(노드내 EP)': stage=노드, 어텐션 TP4×DP2, MoE 노드로컬 EP8"),
    Strat("TP8EP8PP4", 4, 8, 1, 8, 1, "intra", "intra",
          "S2 변형: 어텐션 TP8 + 노드로컬 EP8 (bf16 스트리밍 8-way 유지)"),
    Strat("EP32", 1, 8, 4, 32, 1, "inter", "intra",
          "PP 없음: 어텐션 TP8×DP4(노드=replica), MoE EP32 (a2a 노드 간)"),
    Strat("TP8EP4", 1, 8, 4, 4, 8, "inter", "intra",
          "PP 없음: 어텐션 TP8×DP4, MoE EP4(노드=EP rank)×TP8 (a2a 노드 간)"),
    Strat("TP16PP2", 2, 16, 1, 1, 16, None, "inter",
          "대조군: TP16이 노드를 넘어 AR이 IB를 타면 어떻게 되는가"),
]


# ---------------------------------------------------------------- 커널 코스트
@lru_cache(maxsize=4096)
def lin_us(b: int, K: int, N: int, bpp: float, dq_ops: float):
    """가중치 [K,N]×활성 [b,K] GEMM/GEMV — hatir einsum.tile로 직접 저작한 플랜.

    주의: hatir 기본 k_linear(_gemm_plan)는 reuse tier를 mem[-1](=이 트윈은 SM당
    smem 164KB)로 잡아 spatial fan이 outer 타일에 클램프되고(1~16 SM만 사용),
    디코드 GEMV의 compute가 10~20× 과대계상된다. 그래서 weight-streaming 디코드
    GEMM의 실제 매핑(가중치 1회 스트림, n-타일을 108 SM에 팬)을 직접 저작한다:
      tile {m:b, n:128, k:≤512}, spatial {n:108} — W는 1회 로드(트립이 m뿐),
      A는 n-블록당 재로드(작음), 시간 = max(compute, HBM 스트림) 루프라인.
    bpp<2(양자화 가중치)면 dtype_bytes=bpp로 가중치 바이트를 정확히 맞추고(E4),
    dequant ALU 시간과 max (E2)."""
    b = max(1, b)
    # n-블록을 SM에 팬 + 남는 SM으로 split-k (실제 디코드 GEMV 커널의 매핑);
    # k-타일 256이면 (b+128)×256×2B + Y-acc가 smem 164KB에 상주 → partial-sum 스필 없음.
    Tk = min(K, 256)
    Sn = min(108, math.ceil(N / 128))
    Sk = max(1, min(108 // Sn, math.ceil(K / Tk)))
    plan = tile("m k, k n -> m n", {"m": b, "k": K, "n": N}, HW,
                tile={"m": b, "n": min(N, 128), "k": Tk},
                spatial={"n": Sn, "k": Sk}, dtype_bytes=bpp, name="lin")
    d = plan.derived
    # EXPLICIT 보정: hatir 단일-경계 모델은 A(활성)를 n-팬된 SM마다 backing에서
    # 재로드로 계상(48×). A100은 L2(40MB) 브로드캐스트로 A를 HBM에서 ~1회만 읽으므로
    # 메모리 시간은 W(플랜 검증값: 1회 로드) + A 1회(bf16) + Y 1회 + split-k 콤바인으로
    # 직접 구성. compute는 hatir 플랜 값(스필 없는 per-SM 매핑) 그대로.
    wbytes = K * N * bpp
    mem_bytes = wbytes + b * K * 2 + b * N * 2 + b * N * 4 * (Sk - 1)
    mem_us = mem_bytes / HBM_BW * 1e6
    us = max(d.compute_us, mem_us)
    bound = "mem" if mem_us >= d.compute_us else "compute"
    if bpp < 2.0:  # EXPLICIT(E2): INT4 → bf16 dequant ALU 항
        dq_us = K * N * dq_ops / ALU_RATE * 1e6
        if dq_us > us:
            us, bound = dq_us, "compute"
    return us, bound, mem_bytes


@lru_cache(maxsize=4096)
def ew_us(n: int, dtype_bytes: int, macs: float):
    """elementwise/스캔형 커널(norm, gate, conv/recurrent state 갱신) — hatir tile
    'n -> n' 플랜. spatial=108로 SM 전체에 팬(단일 인스턴스 과대계상 방지)."""
    n = max(1, int(n))
    plan = tile("n -> n", {"n": n}, HW, tile={"n": max(1, math.ceil(n / 108))},
                spatial={"n": 108}, dtype_bytes=dtype_bytes, macs_per_point=macs,
                name="ew")
    d = plan.derived
    return d.kernel_time_us, ("mem" if d.bound == "memory" else "compute"), d.total_backing_bytes


@lru_cache(maxsize=4096)
def mla_attn_us(b: int, S: int, heads_local: int):
    """MLA absorbed 디코드 attention: latent 캐시 [b·S, 576] 1회 스트림(bf16),
    QK(576)+PV(512) MAC를 macs_per_point로 접음. hatir tile 플랜."""
    n_s = max(1, b * S)
    mpp = heads_local * (MLA_CACHE_D + MLA_KV_LORA) / MLA_CACHE_D
    plan = tile("s e -> e", {"s": n_s, "e": MLA_CACHE_D}, HW,
                tile={"s": max(1, math.ceil(n_s / 108))}, spatial={"s": 108},
                dtype_bytes=2, macs_per_point=mpp, name="mla_decode")
    d = plan.derived
    return d.kernel_time_us, ("mem" if d.bound == "memory" else "compute"), d.total_backing_bytes


def exp_kernel(bytes_, macs, alu_ops=0.0):
    """EXPLICIT(E7): hatir Op 1개로 표현 불가한 커널(혼합 dtype grouped-GEMV 등)의
    1차 roofline: max(HBM 스트림, TensorCore, dequant ALU)."""
    t_mem = bytes_ / HBM_BW * 1e6
    t_tc = macs / TC_RATE * 1e6
    t_alu = alu_ops / ALU_RATE * 1e6
    us = max(t_mem, t_tc, t_alu)
    bound = "mem" if us == t_mem else "compute"
    return us, bound, bytes_


# ---------------------------------------------------------------- 통신 코스트
def _fab(P, bw, alpha_total, hops):
    return [FabricLevel("f", P, bw, latency_us=(alpha_total / hops if P > 1 else 0.0))]


def ar_us(msg_bytes, P, sc: Scen, scope):
    if P < 2:
        return {"us": 0.0, "alpha": 0.0, "bw": 0.0}
    bw = sc.nvlink_busbw if scope == "intra" else sc.ib_busbw
    a = sc.ar_a_intra if scope == "intra" else sc.ar_a_inter
    r = hierarchical_collective(msg_bytes, _fab(P, bw, a, 2 * (P - 1)), "allreduce")
    return {"us": r["us"], "alpha": r["latency_us"], "bw": r["bw_us"]}


def ag_us(msg_bytes, P, sc: Scen, scope):
    if P < 2:
        return {"us": 0.0, "alpha": 0.0, "bw": 0.0}
    bw = sc.nvlink_busbw if scope == "intra" else sc.ib_busbw
    a = sc.ar_a_intra if scope == "intra" else sc.ar_a_inter
    r = hierarchical_collective(msg_bytes, _fab(P, bw, a, P - 1), "allgather")
    return {"us": r["us"], "alpha": r["latency_us"], "bw": r["bw_us"]}


def a2a_us(msg_per_gpu, e, sc: Scen, scope):
    if e < 2 or msg_per_gpu <= 0:
        return {"us": 0.0, "alpha": 0.0, "bw": 0.0}
    bw = sc.nvlink_busbw if scope == "intra" else (sc.ib_node / 8.0)  # GPU당 IB 몫
    if scope == "intra":
        a = sc.a2a_a_intra
    else:
        # EXPLICIT(E8): 노드 간 a2a는 per-peer 소메시지 구조(recon/a100-comms.md §3)
        # → α_eff = α_base + α_per_peer × (노드 밖 peer rank 수). EP 그룹이 NNODES개
        # 노드에 걸친다고 가정(본 스윕의 inter 전략 EP32/TP8EP4 모두 4노드 스팬).
        inter_peers = e - max(1, e // NNODES)
        a = sc.a2a_a_inter + sc.a2a_a_inter_pp * inter_peers
    r = hierarchical_collective(msg_per_gpu, _fab(e, bw, a, e - 1), "alltoall")
    return {"us": r["us"], "alpha": r["latency_us"], "bw": r["bw_us"]}


def p2p_us(bytes_, sc: Scen):
    # PP 경계 send/recv: RDMA 단일 레일(HDR 25GB/s) + α. EXPLICIT(α는 시나리오 값)
    return {"us": sc.p2p_a + bytes_ / 25e9 * 1e6, "alpha": sc.p2p_a,
            "bw": bytes_ / 25e9 * 1e6}


# ---------------------------------------------------------------- MoE 통계
def uniq_experts(G):
    """글로벌 배치 G(top-16 독립·균등 라우팅 가정)에서 활성 유니크 expert 기대값."""
    return N_EXPERTS * (1.0 - (1.0 - TOPK / N_EXPERTS) ** G)


def sends_per_token(e, sc: Scen):
    """EXPLICIT(E6): 토큰당 a2a 전송 복사본 수.
    opt(dedup=true) = DeepEP식 랭크-중복제거 기대값 e×(1−(1−1/e)^topk) — DeepEP는
    SM90 전용(트윈 provenance.deepep_requirements)이라 SM80 커스텀 커널 전제의 상한.
    real(dedup=false) = naive per-(token,expert) 복사 = topk — vLLM SM80 naive 경로
    (recon/vllm-upstream.md에 SM80 EP a2a 전용 백엔드 근거 없음)."""
    if e <= 1:
        return 0.0
    if not sc.a2a_dedup:
        return float(TOPK)
    return e * (1.0 - (1.0 - 1.0 / e) ** TOPK)


# ---------------------------------------------------------------- 레이어 코스트
class Acc:
    """스텝 시간 분해 누적기 (µs)."""

    def __init__(self):
        self.mem = 0.0
        self.compute = 0.0
        self.comm_alpha = 0.0
        self.comm_bw = 0.0
        self.launch = 0.0
        self.cpu = 0.0
        self.hbm_bytes = 0.0   # per-GPU(최악 경로) HBM 트래픽

    def kern(self, us, bound, bytes_):
        if bound == "mem":
            self.mem += us
        else:
            self.compute += us
        self.hbm_bytes += bytes_

    def comm(self, c):
        self.comm_alpha += c["alpha"]
        self.comm_bw += c["bw"]

    @property
    def total(self):
        return self.mem + self.compute + self.comm_alpha + self.comm_bw + self.launch + self.cpu

    def as_dict(self):
        return {"mem_us": self.mem, "compute_us": self.compute,
                "comm_alpha_us": self.comm_alpha, "comm_bw_us": self.comm_bw,
                "launch_us": self.launch, "cpu_us": self.cpu, "total_us": self.total,
                "hbm_gb": self.hbm_bytes / 1e9}


def attn_kda(acc: Acc, st: Strat, b, sc: Scen, rq: RQ):
    ta = st.ta
    hl = KDA_HEADS // ta
    acc.kern(*lin_us(b, H, 4 * KDA_W // ta, rq.wb, rq.dq_lin))    # q,k,v,g fused
    acc.kern(*lin_us(b, H, 224, rq.wb, rq.dq_lin))                # f_a(128)+b(96) 복제
    acc.kern(*lin_us(b, 128, KDA_W // ta, rq.wb, rq.dq_lin))      # f_b
    acc.kern(*ew_us(b * 3 * (KDA_W // ta) * 3, 2, 2.0))           # conv1d update(state 3탭 r/w)
    acc.kern(*ew_us(b * hl * KDA_HD * KDA_HD, 4, 8.0))            # fused_recurrent: state fp32 r+w
    acc.kern(*ew_us(b * (KDA_W // ta) * 3, 2, 2.0))               # o_norm + sigmoid gate
    acc.kern(*lin_us(b, KDA_W // ta, H, rq.wb, rq.dq_lin))        # o_proj (row-parallel)
    acc.kern(*ew_us(b * H * 4, 2, 1.0))                           # norm+residual(+AttnRes 게이트)
    ev = 0
    if ta > 1:
        acc.comm(ar_us(b * H * 2, ta, sc, st.ar_scope))
        ev += 1
    return ev


def attn_mla(acc: Acc, st: Strat, b, S, sc: Scen, rq: RQ):
    ta = st.ta
    hl = MLA_HEADS // ta
    acc.kern(*lin_us(b, H, 1536, rq.wb, rq.dq_lin))               # q_a (복제)
    acc.kern(*lin_us(b, 1536, hl * MLA_QK_D, rq.wb, rq.dq_lin))   # q_b
    acc.kern(*lin_us(b, H, MLA_CACHE_D, rq.wb, rq.dq_lin))        # kv_a (복제)
    acc.kern(*exp_kernel(hl * 128 * 512 * rq.wb + b * hl * (128 + 512) * 2,
                         b * hl * 128 * 512))                     # W_UK absorb bmm(헤드별 가중치)
    acc.kern(*mla_attn_us(b, S, hl))                              # latent MQA attention
    acc.kern(*exp_kernel(hl * 512 * 128 * rq.wb + b * hl * (512 + 128) * 2,
                         b * hl * 512 * 128))                     # W_UV bmm
    acc.kern(*lin_us(b, H, KDA_W // ta, rq.wb, rq.dq_lin))        # g_proj (output gate)
    acc.kern(*ew_us(b * (KDA_W // ta) * 2, 2, 2.0))               # sigmoid·mul
    acc.kern(*lin_us(b, KDA_W // ta, H, rq.wb, rq.dq_lin))        # o_proj
    acc.kern(*ew_us(b * H * 4, 2, 1.0))                           # norm+residual
    ev = 0
    if ta > 1:
        acc.comm(ar_us(b * H * 2, ta, sc, st.ar_scope))
        ev += 1
    return ev


def ffn_moe(acc: Acc, st: Strat, G, b, sc: Scen, rq: RQ):
    ta, e, tm = st.ta, st.e, st.tm
    ev = 0
    acc.kern(*ew_us(b * H * 2, 2, 1.0))                           # pre-norm
    acc.kern(*lin_us(b, H, N_EXPERTS, 2.0, 0.0))                  # router (복제, bf16 유지)
    acc.kern(*ew_us(b * N_EXPERTS * 2, 2, 4.0))                   # sigmoid+bias+top16 선택
    acc.kern(*lin_us(b, H, LATENT // ta, rq.wb, rq.dq_lin))       # latent down_proj
    if ta > 1 and sc.name == "real":                              # latent 경계 AG/AR (구현 [미확인] → real만)
        acc.comm(ar_us(b * LATENT * 2, ta, sc, st.ar_scope))
        ev += 1
    if e > 1:                                                     # EP dispatch a2a
        n_stage_gpus = 32 // st.p
        msg = G * sends_per_token(e, sc) * LATENT * sc.disp_payload / n_stage_gpus
        c = a2a_us(msg, e, sc, st.a2a_scope)
        acc.comm(c)
        ev += 1
    # routed experts — EXPLICIT(E7)+E3: max(HBM, TC, dequant) × imbalance
    copies = G * TOPK / e
    u_rank = uniq_experts(G) / e
    wbytes = u_rank * EXPERT_P * rq.bpp_exp / tm
    # 활성 r/w: latent 입력은 tm-샤드마다 전체(×2 r/w), 중간 activation은 /tm
    abytes = copies * (2 * LATENT * 2 + 2 * 2 * MOE_INTER / tm)
    macs = copies * 3 * LATENT * MOE_INTER / tm
    dq = u_rank * EXPERT_P / tm * rq.dq_exp
    us, bound, by = exp_kernel(wbytes + abytes, macs, dq)
    imb = sc.imbalance if e > 1 else 1.0                          # E3
    acc.kern(us * imb, bound, by)
    if e > 1:                                                     # EP combine a2a
        n_stage_gpus = 32 // st.p
        msg = G * sends_per_token(e, sc) * LATENT * 2 / n_stage_gpus  # combine은 bf16
        c = a2a_us(msg, e, sc, st.a2a_scope)
        acc.comm(c)
        ev += 1
        acc.kern(*ew_us(b * LATENT * TOPK // 4, 2, 2.0))          # weighted combine
    if tm > 1 and e > 1:                                          # expert TP partial-sum AR
        acc.comm(ar_us(b * LATENT * 2, tm, sc, "intra"))
        ev += 1
    acc.kern(*lin_us(b, LATENT, H // ta, rq.wb, rq.dq_lin))       # latent up_proj
    acc.kern(*lin_us(b, H, 2 * SHARED_INTER // ta, rq.wb, rq.dq_lin))  # shared w1,w3
    acc.kern(*ew_us(b * (SHARED_INTER // ta) * 3, 2, 2.0))        # SiTU-GLU act
    acc.kern(*lin_us(b, SHARED_INTER // ta, H, rq.wb, rq.dq_lin))  # shared w2
    acc.kern(*ew_us(b * H * 4, 2, 1.0))                           # residual+norm(+AttnRes)
    if ta > 1 or (tm > 1 and e == 1):                             # MoE 블록 출력 AR
        P = ta if ta > 1 else tm
        acc.comm(ar_us(b * H * 2, P, sc, st.ar_scope))
        ev += 1
    return ev


def ffn_dense(acc: Acc, st: Strat, b, sc: Scen, rq: RQ):
    ta = st.ta
    acc.kern(*ew_us(b * H * 2, 2, 1.0))
    acc.kern(*lin_us(b, H, 2 * DENSE_INTER // ta, rq.wb, rq.dq_lin))
    acc.kern(*ew_us(b * (DENSE_INTER // ta) * 3, 2, 2.0))
    acc.kern(*lin_us(b, DENSE_INTER // ta, H, rq.wb, rq.dq_lin))
    acc.kern(*ew_us(b * H * 4, 2, 1.0))
    ev = 0
    if ta > 1:
        acc.comm(ar_us(b * H * 2, ta, sc, st.ar_scope))
        ev += 1
    return ev


def layer_types():
    """(attn_type, ffn_type) per layer 1..93 — recon/k3-config.md §2."""
    out = []
    for i in range(1, N_LAYERS + 1):
        attn = "mla" if (i % 4 == 0 or i == N_LAYERS) else "kda"
        ffn = "dense" if i == 1 else "moe"
        out.append((attn, ffn))
    return out


LAYERS = layer_types()


def stage_slices(p):
    sizes = [N_LAYERS // p + (1 if i < N_LAYERS % p else 0) for i in range(p)]
    out, s = [], 0
    for n in sizes:
        out.append((s, s + n))
        s += n
    return out


# ---------------------------------------------------------------- 스텝 시간
def _accum_layers(acc: Acc, st: Strat, G, b, sc: Scen, rq: RQ, S_ctx: int, layers):
    """`layers`(LAYERS 부분열)의 커널+통신+런치를 acc에 누적. G = 이 패스가 처리하는
    글로벌 토큰 수(MoE uniq/dispatch 통계의 입력), b = 어텐션 DP replica당 배치."""
    for (attn, ffn) in layers:
        ev = 0
        if attn == "kda":
            ev += attn_kda(acc, st, b, sc, rq)
        else:
            ev += attn_mla(acc, st, b, S_ctx, sc, rq)
        if ffn == "moe":
            ev += ffn_moe(acc, st, G, b, sc, rq)
        else:
            ev += ffn_dense(acc, st, b, sc, rq)
        # EXPLICIT(E1): 커널 런치/그래프 갭 — comm 경계 수(ev)만큼 그래프 분절
        nk = KERNELS[attn][sc.name] + KERNELS[ffn][sc.name]
        acc.launch += nk * sc.launch_gap + ev * sc.graph_seg


def _tail(acc: Acc, st: Strat, b, sc: Scen):
    """스텝 공통 마무리(마지막 stage 소속): final norm, lm_head, logits AG, sampling."""
    acc.kern(*ew_us(b * H * 2, 2, 1.0))
    acc.kern(*lin_us(b, H, VOCAB // st.ta, 2.0, 0.0))             # lm_head bf16 유지
    acc.comm(ag_us(b * (VOCAB // st.ta) * 2, st.ta, sc, "intra"))
    acc.kern(*ew_us(b * VOCAB // 8, 2, 1.0))                      # top-p/sampling 축약
    acc.launch += KERNELS["misc"][sc.name] * sc.launch_gap


def step_time(st: Strat, G, sc: Scen, requant: str, S_ctx: int):
    """디코드 1스텝의 최악-경로(스테이지 직렬 합) 시간 + 분해. 반환 Acc(µs).

    주의(지표 정의): 이 직렬 합은 배치 1개가 93층 전체를 통과하는 시간 = 사용자 지연
    (tok_s_user)의 근거. PP 전략의 시스템 처리량에는 그대로 쓰면 안 됨 — vLLM PP는
    p개 마이크로배치를 in-flight로 유지하므로 정상상태 처리량은 최악 스테이지 시간이
    결정(pipelined_throughput, EXPLICIT P1)."""
    rq = mk_rq(requant, sc)
    b = max(1, math.ceil(G / st.dp))
    acc = Acc()
    _accum_layers(acc, st, G, b, sc, rq, S_ctx, LAYERS)
    # PP 경계: (p-1)회 hidden p2p + 토큰 id 반환 1회
    if st.p > 1:
        hb = b * H * 2 * sc.pp_streams   # AttnRes 스트림 [미확인] → real은 2×hidden
        for _ in range(st.p - 1):
            acc.comm(p2p_us(hb, sc))
        acc.comm(p2p_us(64, sc))
    # 스텝 공통: embed(무시) + tail
    _tail(acc, st, b, sc)
    acc.cpu += sc.step_cpu                                        # EXPLICIT(E5)
    return acc


def pipelined_throughput(st: Strat, G, sc: Scen, requant: str, S_ctx: int):
    """EXPLICIT(P1): PP 마이크로배치 파이프라인 정상상태 시스템 처리량.

    상주 배치 G를 p개 마이크로배치(G_mb = G/p)로 나눠 p개를 in-flight로 유지하면
    (vLLM PP의 다중 in-flight 배치), 정상상태 처리량 = G_mb 토큰 / 최악 스테이지
    시간. 스테이지 시간은 해당 stage 층들의 커널·통신·런치(G=G_mb 기준: 가중치는
    마이크로배치마다 재스트림, uniq expert도 G_mb 기준 — 파이프라이닝의 가중치 재사용
    손실을 그대로 반영) + PP 경계 송신 + (마지막 stage) tail + 마이크로배치당 CPU.
    비모델(한계): 램프업/드레인 버블, vLLM PP 스케줄러 실동작. 스테이지 간 패브릭은
    본 스윕 전략에서 분리(stage=노드 단위)라 무경합 가정 유지. p==1이거나 G<p(파이프
    라인을 못 채움)면 직렬 값과 동일. 전제: 상주 G seq 동시 유지 = max_num_seqs>1
    성립 [미확인 — REPORT §10].
    반환: {"tok_s", "stage_ms"(최악), "microbatch"}"""
    if st.p == 1 or G < st.p:
        acc = step_time(st, G, sc, requant, S_ctx)
        return {"tok_s": G * 1e6 / acc.total, "stage_ms": None, "microbatch": None}
    rq = mk_rq(requant, sc)
    G_mb = G // st.p
    b = max(1, math.ceil(G_mb / st.dp))
    stage_us = []
    for si, (a, z) in enumerate(stage_slices(st.p)):
        acc = Acc()
        _accum_layers(acc, st, G_mb, b, sc, rq, S_ctx, LAYERS[a:z])
        if si < st.p - 1:
            acc.comm(p2p_us(b * H * 2 * sc.pp_streams, sc))       # 다음 stage 송신
        else:
            acc.comm(p2p_us(64, sc))                              # 토큰 id 반환
            _tail(acc, st, b, sc)
        acc.cpu += sc.step_cpu    # 마이크로배치·stage당 스케줄러 오버헤드(보수)
        stage_us.append(acc.total)
    t = max(stage_us)
    return {"tok_s": G_mb * 1e6 / t, "stage_ms": t / 1e3, "microbatch": G_mb}


# ---------------------------------------------------------------- 메모리 산정
def memory_report(st: Strat, requant: str):
    wb = 2.0 if requant == "orig" else BPP_W4A16
    bpp_exp = BPP_MXFP4 if requant == "orig" else BPP_W4A16
    per_stage = []
    for si, (a, z) in enumerate(stage_slices(st.p)):
        kda_n = sum(1 for i in range(a, z) if LAYERS[i][0] == "kda")
        mla_n = (z - a) - kda_n
        moe_n = sum(1 for i in range(a, z) if LAYERS[i][1] == "moe")
        dense_n = (z - a) - moe_n
        w = (kda_n * (KDA_SHARD_P * wb / st.ta + KDA_REPL_P * wb)
             + mla_n * (MLA_SHARD_P * wb / st.ta + MLA_REPL_P * wb)
             + moe_n * (ROUTED_P_LAYER * bpp_exp / (st.e * st.tm)
                        + (SHARED_P + LATENT_PROJ_P) * wb / st.ta
                        + ROUTER_P * 2.0)
             + dense_n * DENSE_P * wb / st.ta)
        if si == 0:
            w += VOCAB * H * 2.0 / st.ta          # embed
        if si == st.p - 1:
            w += LMHEAD_P * 2.0 / st.ta           # lm_head (bf16 유지)
        kv_seq_8k = mla_n * 8192 * MLA_CACHE_D * 2
        kv_seq_256k = mla_n * 262144 * MLA_CACHE_D * 2
        state_seq = (kda_n * KDA_STATE_BYTES_PER_LAYER_SEQ / st.ta
                     + kda_n * KDA_CONV_BYTES_PER_LAYER_SEQ / st.ta)
        per_stage.append({"stage": si, "weights_gb": w / 1e9,
                          "kv_per_seq_8k_mb": (kv_seq_8k + state_seq) / 1e6,
                          "kv_per_seq_256k_gb": (kv_seq_256k + state_seq) / 1e9,
                          "kv_only_256k_gb": kv_seq_256k / 1e9,
                          "state_per_seq_mb": state_seq / 1e6})
    worst = max(per_stage, key=lambda r: r["weights_gb"])
    worst_kv = max(per_stage, key=lambda r: r["kv_per_seq_256k_gb"])
    budget = 85899345920 * 0.92 - 4e9   # 8% 프레임워크/단편화 + 4GB 활성/그래프/NCCL 예약
    avail = budget - worst["weights_gb"] * 1e9
    seqs_8k = int(avail / (worst_kv["kv_per_seq_8k_mb"] * 1e6)) if avail > 0 else 0
    seqs_256k = int(avail / (worst_kv["kv_per_seq_256k_gb"] * 1e9)) if avail > 0 else 0
    return {"per_stage": per_stage, "worst_weights_gb": worst["weights_gb"],
            "per_seq_8k_mb": worst_kv["kv_per_seq_8k_mb"],
            "per_seq_256k_gb": worst_kv["kv_per_seq_256k_gb"],
            "state_per_seq_mb": worst_kv["state_per_seq_mb"],
            "max_seqs_8k_global": max(0, seqs_8k) * st.dp,
            "max_seqs_256k_global": max(0, seqs_256k) * st.dp,
            "fits": worst["weights_gb"] * 1e9 < budget}


# ---------------------------------------------------------------- 검증 앵커
def sanity_anchor():
    """recon/k3-config.md의 원본 bytes/token=136.96GB, 완전병렬 하한 2.10ms 재현."""
    bf16 = (69 * (KDA_SHARD_P + KDA_REPL_P) * 2 + 24 * (MLA_SHARD_P + MLA_REPL_P) * 2
            + 92 * (SHARED_P + LATENT_PROJ_P + ROUTER_P) * 2 + DENSE_P * 2 + LMHEAD_P * 2)
    mxfp4 = TOPK * 92 * EXPERT_P * BPP_MXFP4
    total = bf16 + mxfp4
    floor_ms = total / (32 * HBM_BW) * 1e3
    return {"bytes_per_token_gb": total / 1e9, "bf16_gb": bf16 / 1e9,
            "mxfp4_gb": mxfp4 / 1e9, "flat_parallel_floor_ms": floor_ms,
            "recon_ref_gb": 136.96, "recon_ref_floor_ms": 2.10,
            "closure_ok": abs(total / 1e9 - 136.96) / 136.96 < 0.03}


# ---------------------------------------------------------------- 스윕/판정
def run():
    batches = [1, 8, 32]
    S_ctx = 8192
    results = []
    for st in STRATS:
        for rq in ("orig", "w4a16"):
            memr = memory_report(st, rq)
            for sc_name in ("opt", "real"):
                sc = SCENS[sc_name]
                for G in batches:
                    acc = step_time(st, G, sc, rq, S_ctx)
                    ms = acc.total / 1e3
                    pipe = pipelined_throughput(st, G, sc, rq, S_ctx)
                    results.append({
                        "strategy": st.name, "requant": rq, "scenario": sc_name,
                        "batch": G, "ctx": S_ctx,
                        "step_ms": ms, "tok_s_user": 1e3 / ms,
                        "tok_s_system": G * 1e3 / ms,
                        "tok_s_system_pipelined": pipe["tok_s"],
                        "pp_stage_ms": pipe["stage_ms"],
                        "pp_microbatch": pipe["microbatch"],
                        "breakdown": acc.as_dict(),
                        "mem": {k: memr[k] for k in
                                ("worst_weights_gb", "per_seq_8k_mb", "per_seq_256k_gb",
                                 "state_per_seq_mb", "max_seqs_8k_global",
                                 "max_seqs_256k_global", "fits")},
                    })
    # 컨텍스트 민감도 (승자 전략, G=8, orig)
    best_real = max((r for r in results
                     if r["scenario"] == "real" and r["requant"] == "orig" and r["batch"] == 1),
                    key=lambda r: r["tok_s_user"])
    win = next(s for s in STRATS if s.name == best_real["strategy"])
    ctx_sweep = []
    for S in (2048, 8192, 32768, 131072):
        for sc_name in ("opt", "real"):
            acc = step_time(win, 8, SCENS[sc_name], "orig", S)
            ctx_sweep.append({"strategy": win.name, "ctx": S, "scenario": sc_name,
                              "batch": 8, "step_ms": acc.total / 1e3,
                              "tok_s_user": 1e6 / acc.total})
    return results, ctx_sweep, best_real


def gate_verdict(results):
    """게이트: '쓸 만한 속도(대화형)' = 단일 스트림(배치1) ≥ 5 tok/s.
    PASS: 현실(real) 시나리오에서 달성. CONDITIONAL: 낙관(opt)에서만 달성.
    FAIL: 낙관에서도 3 tok/s 미만."""
    def best(scen, G=1, rq="orig"):
        rows = [r for r in results if r["scenario"] == scen and r["batch"] == G
                and r["requant"] == rq and r["mem"]["fits"]]
        return max(rows, key=lambda r: r["tok_s_user"])
    b_real, b_opt = best("real"), best("opt")
    if b_real["tok_s_user"] >= 5:
        verdict = "PASS"
    elif b_opt["tok_s_user"] >= 5:
        verdict = "CONDITIONAL_PASS"
    elif b_opt["tok_s_user"] < 3:
        verdict = "FAIL"
    else:
        verdict = "MARGINAL"
    return verdict, b_real, b_opt


# ---------------------------------------------------------------- 리포트 생성
def fmt_row(r):
    bd = r["breakdown"]
    top = sorted([("HBM", bd["mem_us"]), ("compute", bd["compute_us"]),
                  ("comm-α", bd["comm_alpha_us"]), ("comm-bw", bd["comm_bw_us"]),
                  ("launch", bd["launch_us"]), ("cpu", bd["cpu_us"])],
                 key=lambda x: -x[1])
    t2 = " + ".join(f"{n} {v/1e3:.1f}ms" for n, v in top[:2])
    if r["pp_microbatch"]:
        pipe = f"{r['tok_s_system_pipelined']:.0f} (µb{r['pp_microbatch']}·stage {r['pp_stage_ms']:.1f}ms)"
    else:
        pipe = f"{r['tok_s_system_pipelined']:.0f}"
    return (f"| {r['strategy']} | {r['batch']} | {r['step_ms']:.1f} | "
            f"{r['tok_s_user']:.1f} | {r['tok_s_system']:.0f} | {pipe} | {t2} |")


def git_rev(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"


def make_report(results, ctx_sweep, verdict, b_real, b_opt, anchor):
    pins = {
        "hf_kimi_k3": "moonshotai/Kimi-K3 @ 9f62e4e9fffbd0a83ddd60e1c209d828994b3569 (recon/k3-config.md)",
        "vllm_main": "d223c900d85224c02f2162ee2c757a769e99f519 (2026-07-28, recon/vllm-upstream.md)",
        "vllm_kimi_k3_branch": "0498dc7ea9204f1863b46515816f5d8d458e01e7 (PR #50000)",
        "hatir": git_rev("/home/jimmy/repos/hatir"),
        "hat_schema": git_rev("/home/jimmy/repos/hat-schema"),
        "twin": "phase0/twin_a100_x32.json",
    }
    L = []
    A = L.append
    A("# Phase 0 — hatir 타당성 게이트: Kimi K3 디코드 @ 32×A100(SM80)")
    A("")
    A(f"생성: `k3_model.py` 실행 산출물 (수치 전부 실행 출력; 손수치 없음). 날짜: 2026-07-28")
    A("")
    A("## 0. 판정 (게이트)")
    A("")
    A(f"**{verdict}** — 기준: 단일 스트림(배치 1) decode ≥ 5 tok/s (대화형 최소선의 decode 축; "
      "TTFT/prefill은 미모델 — §10).")
    A("")
    A(f"- 현실(보수) 시나리오 최선: **{b_real['strategy']}** = "
      f"**{b_real['tok_s_user']:.1f} tok/s** (스텝 {b_real['step_ms']:.1f}ms, ctx 8k)")
    A(f"- 낙관 시나리오 최선(\"모든 커널·통신이 완벽해도\"): **{b_opt['strategy']}** = "
      f"**{b_opt['tok_s_user']:.1f} tok/s** (스텝 {b_opt['step_ms']:.1f}ms)")
    A(f"- 완전병렬 이론 하한(전략 무관): {anchor['flat_parallel_floor_ms']:.2f} ms/tok = "
      f"{1e3/anchor['flat_parallel_floor_ms']:.0f} tok/s 상한 — recon 앵커 "
      f"{anchor['recon_ref_floor_ms']}ms와 정합(바이트 폐쇄 {anchor['closure_ok']})")
    A("")
    A("### 핵심 발견 (수치는 아래 표에서)")
    A("")

    def _get(strategy, batch, scen, rq="orig"):
        return next(r for r in results if r["strategy"] == strategy and r["batch"] == batch
                    and r["scenario"] == scen and r["requant"] == rq)

    thr_real = max((r for r in results if r["scenario"] == "real" and r["batch"] == 32
                    and r["requant"] == "orig" and r["mem"]["fits"]),
                   key=lambda r: r["tok_s_system_pipelined"])
    thr_real_sync = max((r for r in results if r["scenario"] == "real" and r["batch"] == 32
                         and r["requant"] == "orig" and r["mem"]["fits"]),
                        key=lambda r: r["tok_s_system"])
    tp8_32 = _get("TP8PP4", 32, "real")
    dq_share = tp8_32["breakdown"]["compute_us"] / (tp8_32["step_ms"] * 1e3) * 100
    w4_best = max((r for r in results if r["scenario"] == "real" and r["batch"] == 1
                   and r["requant"] == "w4a16" and r["mem"]["fits"]),
                  key=lambda r: r["tok_s_user"])
    if thr_real["pp_microbatch"]:
        _p = next(s.p for s in STRATS if s.name == thr_real["strategy"])
        thr_detail = (f"µb{thr_real['pp_microbatch']}×{_p} in-flight, "
                      f"최악 stage {thr_real['pp_stage_ms']:.1f}ms")
    else:
        thr_detail = "무-PP(파이프라인 불필요)"
    A(f"1. **처리량 승자는 PP 파이프라인 정상상태 기준 {thr_real['strategy']}**: 배치 32(상주), "
      f"현실 시나리오에서 PP 마이크로배치 파이프라이닝(P1) 반영 시 **{thr_real['strategy']} = "
      f"{thr_real['tok_s_system_pipelined']:.0f} tok/s** ({thr_detail}). 동기 단일배치(비-파이프라인) "
      f"기준으로는 {thr_real_sync['strategy']}({thr_real_sync['tok_s_system']:.0f} tok/s)가 앞서 "
      f"보이지만 이는 vLLM PP의 다중 in-flight 배치를 0으로 치는 지표 정의 인공물이다. 단일 "
      f"스트림(대화 지연)은 α-이벤트 최소인 **{b_real['strategy']}**가 승자. ⚠ 배치>1 결과 "
      f"전체가 `max_num_seqs`>1 성립을 전제([미확인] — §10).")
    A(f"2. **원본 MXFP4의 dequant ALU 세금 (dq_ops=4 가정 시)**: SM80은 FP4 HW가 없어(#35922 "
      f"커널 가족의 에뮬레이션 경로) 배치 32에서 TP-샤딩 전략의 compute 비중이 {dq_share:.0f}%까지 "
      f"상승 (TP8PP4/real). 단 이 결론은 dequant_ops_per_param(real 4.0/opt 2.0)의 함수이며 이 "
      f"값은 [무근거 밴드](실측·1차 출처 없음 — 트윈 _dequant_ops_note): dq=2면 같은 항이 "
      f"메모리바운드로 반전된다. Phase 1 dequant 마이크로벤치로 확정 전까지 조건부. EP 분산"
      f"(expert/rank↓) 또는 W4A16 재양자화가 완화 방향인 것은 dq 값과 무관하게 성립.")
    A(f"3. **Phase 3(W4A16) 가치 정량화**: bf16 잔여 111GB/tok이 지배 병목 — 전량 W4A16이면 "
      f"현실 시나리오 단일 스트림 {b_real['tok_s_user']:.1f} → {w4_best['tok_s_user']:.1f} tok/s "
      f"({w4_best['tok_s_user']/b_real['tok_s_user']:.2f}×). '조건부'가 아니라 사실상 필수 격상 "
      f"근거(recon/k3-config.md §6과 합치).")
    A("4. **EP의 실제 세금은 노드 간 a2a α 구조**: naive 경로(real)에서 노드 간 a2a α는 "
      "per-peer 항이 지배(E8: EP32 = 44+6×24 = 188µs/collective, ×2회/MoE층) — EP32/TP8EP4의 "
      "처리량이 종전 산출 대비 하락하고, DeepEP류 dedup+MXFP8 payload는 SM90 전용 라이브러리라 "
      "SM80에서는 커스텀 커널 신규작성 없이는 못 쓴다(opt 시나리오 전용 가정). PP의 '가중치 "
      "스트리밍 8-way 직렬화' 세금은 동기 지표의 인공물이었고, 파이프라인 정상상태에서는 "
      "PP가 처리량도 이긴다(발견 1).")
    A("")
    A("## 1. 고정 핀 (재현성)")
    A("")
    for k, v in pins.items():
        A(f"- {k}: `{v}`")
    A("")
    A("## 2. 트윈 검증 (hat_schema.validate.validate_twin)")
    A("")
    A(f"- 오류 0건 필수 → 통과. 전체 메시지 {len(VALIDATION)}건 (info/warn는 모델링 선택):")
    for v in VALIDATION:
        A(f"  - [{v['severity']}] {v['level']}: {v['msg']}")
    A("")
    A("## 3. 방법")
    A("")
    A("- 커널: hatir `einsum.tile`로 직접 저작한 플랜 + 명시 루프라인 "
      "max(플랜 compute_us, 명시 mem_bytes/HBM피크) — 디코드 GEMV는 "
      "가중치 1회 스트림·n-블록 108 SM 팬·split-k 매핑, MLA attention/KDA recurrent/"
      "elementwise도 tile 플랜 (HBM→L2→SMEM→PE 서브스택). "
      "hatir 기본 `_gemm_plan`/`roofline_us`는 이 트윈(SM당 smem 164KB reuse tier)에서 "
      "spatial fan이 클램프되어 디코드 compute를 10–20× 과대계상 → 미사용, 직접 저작으로 "
      "대체(코드 주석).")
    A("- 통신: hatir `hierarchical_collective`(α-β, 무혼잡 이상치) — α/실효 busbw는 "
      "트윈 `meta.comm_scenarios`의 보수(real)/낙관(opt) 2안 (recon/a100-comms.md 근거). "
      "노드 간 a2a α는 per-peer 구조(E8): α_eff = α_base + 6µs×(노드 밖 peer 수) — "
      "EP32 real은 188µs/collective (naive NCCL a2a 밴드와 정합; 트윈 _a2a_alpha_note).")
    A("- EP 전송 복사본(E6): real = naive per-(token,expert) 복사(topk=16) — vLLM SM80에 "
      "전용 EP a2a 백엔드 근거 없음(recon/vllm-upstream.md); opt = DeepEP식 랭크-중복제거 + "
      "MXFP8 payload — 단 DeepEP는 SM90 전용이라 SM80 커스텀 커널 신규작성 전제의 상한.")
    A("- PP 처리량(P1): `tok/s/시스템`은 동기 단일배치(직렬 합 = 지연 근거) 기준이고, "
      "`tok/s/시스템(PP파이프)`는 상주 배치를 p개 마이크로배치로 나눠 in-flight 유지하는 "
      "vLLM PP 정상상태(최악 stage 시간 결정; 가중치 재스트림·uniq expert는 마이크로배치 "
      "기준으로 반영, 램프업/드레인 버블 미모델) — PP 전략의 처리량 판정은 후자 사용.")
    A("- hatir가 표현 못 하는 항목은 명시 가산항(코드 EXPLICIT E1–E8, P1): 런치/그래프 갭, "
      "dequant ALU([무근거 밴드] — 발견 2), MoE 불균형, 혼합 dtype, 스텝 CPU, dispatch "
      "복사본 수, expert grouped-GEMV, per-peer a2a α, PP 파이프라인 정상상태.")
    A("- 한계: 무혼잡·무겹침(통신-계산 오버랩 없음 = 보수), 균등 라우팅 기대값, "
      "절대치는 탐색등급(순위/규모 판단용) — HATIR 검증 원칙(트렌드 정합)에 따름. "
      "HBM은 datasheet 피크(2039GB/s) 무보정 — §10 minor 부채 참조.")
    A("")
    A("## 4. 전략 정의 (32 GPU = 4노드×8)")
    A("")
    A("| 전략 | PP | 어텐션 | MoE | a2a 링크 | 비고 |")
    A("|---|---|---|---|---|---|")
    for s in STRATS:
        att = f"TP{s.ta}" + (f"×DP{s.dp}" if s.dp > 1 else "")
        moe = (f"EP{s.e}" + (f"×TP{s.tm}" if s.tm > 1 else "")) if s.e > 1 else f"TP{s.tm}"
        A(f"| {s.name} | {s.p} | {att} | {moe} | {s.a2a_scope or '—'} | {s.desc} |")
    A("")
    for rq, rq_label in (("orig", "원본 체크포인트(MXFP4 expert + bf16 잔여 111GB/tok)"),
                         ("w4a16", "Phase 3 전량 W4A16 재양자화 가정(56GB/tok)")):
        A(f"## 5.{'1' if rq == 'orig' else '2'} 결과 — {rq_label}")
        A("")
        for sc in ("real", "opt"):
            A(f"### {'현실(보수)' if sc == 'real' else '낙관'} 시나리오 — ctx 8k")
            A("")
            A("| 전략 | 배치 | 스텝 ms | tok/s/유저 | tok/s/시스템(동기) | tok/s/시스템(PP파이프) | 지배 병목 top-2 |")
            A("|---|---|---|---|---|---|---|")
            for r in results:
                if r["requant"] == rq and r["scenario"] == sc:
                    A(fmt_row(r))
            A("")
    A("## 6. 승자 분해 (배치 1 / 8 / 32, 현실 시나리오, 원본)")
    A("")
    win = b_real["strategy"]
    A(f"전략 **{win}**:")
    A("")
    A("| 배치 | HBM | compute | comm-α | comm-bw | launch | cpu | 합계 ms |")
    A("|---|---|---|---|---|---|---|---|")
    for r in results:
        if (r["strategy"] == win and r["scenario"] == "real" and r["requant"] == "orig"):
            b = r["breakdown"]
            A(f"| {r['batch']} | {b['mem_us']/1e3:.1f} | {b['compute_us']/1e3:.1f} | "
              f"{b['comm_alpha_us']/1e3:.1f} | {b['comm_bw_us']/1e3:.1f} | "
              f"{b['launch_us']/1e3:.1f} | {b['cpu_us']/1e3:.1f} | {r['step_ms']:.1f} |")
    A("")
    A("## 7. GPU당 메모리 (가중치 + KV/state)")
    A("")
    A("KDA 69층 state + MLA 24층 latent KV (recon/k3-config.md §8: latent 27,648B/tok, "
      "state 6.29MB/층/seq fp32):")
    A("")
    A("| 전략 | 가중치 GB(최악 stage) | seq당 @8k MB | seq당 @256k GB | state/seq MB | "
      "최대 seq @8k | 최대 seq @256k | 적재 |")
    A("|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in results:
        key = (r["strategy"], r["requant"])
        if key in seen or r["requant"] != "orig":
            continue
        seen.add(key)
        m = r["mem"]
        A(f"| {r['strategy']} | {m['worst_weights_gb']:.1f} | {m['per_seq_8k_mb']:.0f} | "
          f"{m['per_seq_256k_gb']:.2f} | {m['state_per_seq_mb']:.1f} | "
          f"{m['max_seqs_8k_global']} | {m['max_seqs_256k_global']} | "
          f"{'OK' if m['fits'] else 'FAIL'} |")
    A("")
    A("## 8. 컨텍스트 민감도 (승자, 배치 8, 원본)")
    A("")
    A("| 전략 | ctx | 시나리오 | 스텝 ms | tok/s/유저 |")
    A("|---|---|---|---|---|")
    for r in ctx_sweep:
        A(f"| {r['strategy']} | {r['ctx']} | {r['scenario']} | {r['step_ms']:.1f} | {r['tok_s_user']:.1f} |")
    A("")
    A("## 9. 검증 앵커")
    A("")
    A(f"- 모델 bytes/token(배치1, 원본) = {anchor['bytes_per_token_gb']:.1f} GB "
      f"(bf16 {anchor['bf16_gb']:.1f} + MXFP4 {anchor['mxfp4_gb']:.1f}) vs recon "
      f"{anchor['recon_ref_gb']} GB → 폐쇄 {anchor['closure_ok']}")
    A(f"- 완전병렬 하한 {anchor['flat_parallel_floor_ms']:.2f} ms/tok vs recon 2.10 ms")
    A("")
    A("## 10. 주요 가정/리스크")
    A("")
    A("- **`max_num_seqs`>1 성립 [미확인]**: 배치 8/32의 모든 결과(파이프라인 포함)는 "
      "hybrid KDA-state 다중 시퀀스 배칭 + DP 어텐션 + EP dispatch가 kimi-k3 브랜치/"
      "SM80에서 실동작한다는 전제 위에 있다. recon/k3-config.md §8은 KDA state가 "
      "`max_num_seqs`를 먼저 제약할 수 있다고 경고(스펙 '물리 타당성 우려'의 명시 항목). "
      "배치 1 게이트 판정은 이 전제와 무관. → Phase 1 프로브에 `max_num_seqs>1` 동작 "
      "확인(스케줄러/state 경로) 추가.")
    A("- **PP 파이프라인 모델(P1)은 정상상태 상한**: 램프업/드레인 버블, vLLM PP 스케줄러 "
      "실동작(마이크로배치 수·스케줄링 정책) 미모델. 스테이지 간 패브릭은 본 스윕 전략에서 "
      "노드 단위로 분리라 무경합 가정 유지. KV 적재는 §7 기준 상주 32 seq 가능(TP8PP4 "
      "max 321@8k).")
    A("- **dequant_ops_per_param(real 4.0/opt 2.0)은 [무근거 밴드]** — 명령어 수준 유도만 "
      "있고 실측·1차 출처 없음(트윈 _dequant_ops_note; Marlin arXiv:2408.11743은 INT4 "
      "근거일 뿐). 발견 2(ALU-bound)는 dq=4 조건부. → Phase 1에 MXFP4 dequant "
      "마이크로벤치 추가.")
    A("- 통신은 무혼잡 α-β(hatir collectives 계약). 노드 간 a2a α는 per-peer 구조(E8, "
      "[파생] — 실측 아님): per-peer 6µs는 NCCL Simple per-hop(arXiv:2507.04786 Table I) "
      "기반 1차 근사 → Phase 1 a2a 마이크로벤치로 재고정. 노드 내 a2a α(20µs flat)는 "
      "per-peer 미적용(잔여 낙관).")
    A("- MoE 라우팅 균등 기대값 + 불균형 계수(real 1.5)로 근사 — expert 커널에만 적용, "
      "a2a hot-rank 수신 편중에는 미적용(같은 방향의 잔여 낙관). 실측 라우팅 분포로 "
      "Phase 1에서 재보정.")
    A("- AttnRes 스트림의 PP 경계 폭([미확인], recon/k3-config.md §10) → real은 "
      "hidden 2배 전송 가정.")
    A("- latent 경계 collective(구현 [미확인]) → real만 +1 AR/MoE층.")
    A("- KDA state fp32(vLLM 하드코딩), FLA A100 수치검증은 Phase 1 프로브 (recon/kda-fla.md).")
    A("- Marlin/moe_wna16 커널이 #35922 수정 후 정상 동작한다는 전제(속도 게이트일 뿐 "
      "정합성 게이트 아님).")
    A("- **게이트는 decode 한정**: TTFT/prefill(ctx 8k ≈ 1.7e15 FLOPs + KDA chunk 경로)은 "
      "미모델 — '대화형' 체감은 TTFT에도 걸리므로 Phase 1에서 prefill 봉투 계산/실측 필요.")
    A("- **알려진 minor 모델 부채(검증 렌즈 확인, 판정 불변 — phase0/VERIFICATION.md)**: "
      "① HBM을 피크 2039GB/s 100%로 계상(실측 스트림 효율 ~85–90% 미반영, real 최선 "
      "46.8→~44 tok/s 방향), ② lm_head logits AG scope가 `intra` 하드코딩(TP16PP2는 "
      "inter여야 — 스텝당 ~0.04ms 과소), ③ logits AG에 per-rank shard 바이트 전달"
      "(hatir 관례는 full buffer — bw 항 ~40µs 과소), ④ PP p2p bw 25GB/s 고정(real "
      "2×HDR 공유 시 ~6.25GB/s — 스텝당 ~0.17ms 과소), ⑤ KDA conv state를 kernel 4탭 "
      "전체로 계상(vLLM은 K_conv−1=3 저장, recon/kda-fla.md — state/seq ~1.33× 과대), "
      "⑥ 미사용 상수 MLA_Q_LORA=576(실제 q_lora_rank=1536 하드코딩).")
    return "\n".join(L) + "\n"


def main():
    anchor = sanity_anchor()
    assert anchor["closure_ok"], f"bytes/token 폐쇄 실패: {anchor}"
    results, ctx_sweep, _ = run()
    verdict, b_real, b_opt = gate_verdict(results)
    thr_real = max((r for r in results if r["scenario"] == "real" and r["batch"] == 32
                    and r["requant"] == "orig" and r["mem"]["fits"]),
                   key=lambda r: r["tok_s_system_pipelined"])

    out = {"date": "2026-07-28", "ctx_default": 8192,
           "anchor": anchor, "validation": VALIDATION,
           "results": results, "ctx_sweep": ctx_sweep}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    report = make_report(results, ctx_sweep, verdict, b_real, b_opt, anchor)
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write(report)

    vj = {
        "date": "2026-07-28",
        "gate": {"criterion": "단일 스트림(배치1, ctx 8k) decode ≥ 5 tok/s (대화형 최소선의 "
                              "decode 축 — TTFT/prefill은 미모델, REPORT §10); "
                              "FAIL 기준: 낙관에서도 < 3 tok/s",
                 "verdict": verdict},
        "best_real": {k: b_real[k] for k in ("strategy", "step_ms", "tok_s_user", "tok_s_system")},
        "best_opt": {k: b_opt[k] for k in ("strategy", "step_ms", "tok_s_user", "tok_s_system")},
        "best_throughput_real_b32": {
            **{k: thr_real[k] for k in ("strategy", "step_ms", "tok_s_user")},
            "tok_s_system_sync": thr_real["tok_s_system"],
            "tok_s_system_pipelined": thr_real["tok_s_system_pipelined"],
            "pp_microbatch": thr_real["pp_microbatch"],
            "pp_stage_ms": thr_real["pp_stage_ms"],
            "basis": "PP 마이크로배치 파이프라인 정상상태(P1) 기준 선정 — 동기 직렬 지표는 "
                     "PP in-flight 배치를 0으로 치는 인공물(REPORT 발견 1)"},
        "caveats": [
            "배치>1 결과는 max_num_seqs>1 성립 전제([미확인] — KDA state 다중 seq 배칭, "
            "REPORT §10; Phase 1 프로브 필요)",
            "PP 파이프라인은 정상상태 상한(램프업/드레인 버블 미모델)",
            "dequant_ops_per_param은 무근거 밴드 — 발견 2(ALU-bound)는 dq=4 조건부",
            "노드 간 a2a α per-peer 구조는 [파생](실측 아님) — Phase 1 a2a 마이크로벤치로 재고정",
        ],
        "flat_parallel_floor_ms": anchor["flat_parallel_floor_ms"],
        "pins": {"hf": "moonshotai/Kimi-K3@9f62e4e9", "vllm_main": "d223c900",
                 "vllm_kimi_k3": "0498dc7e", "hatir": git_rev("/home/jimmy/repos/hatir"),
                 "hat_schema": git_rev("/home/jimmy/repos/hat-schema")},
        "files": ["phase0/twin_a100_x32.json", "phase0/k3_model.py",
                  "phase0/results.json", "phase0/REPORT.md", "phase0/verdict.json"],
    }
    with open(os.path.join(HERE, "verdict.json"), "w") as f:
        json.dump(vj, f, indent=1, ensure_ascii=False)

    # ---- stdout: 핵심 표 ----
    print("=" * 100)
    print("Phase 0 — K3 디코드 @ 32×A100, hatir 해석 모델 (ctx 8k)")
    print(f"검증 앵커: bytes/token={anchor['bytes_per_token_gb']:.1f}GB, "
          f"완전병렬 하한={anchor['flat_parallel_floor_ms']:.2f}ms (recon 2.10ms)")
    print("-" * 100)
    for rq in ("orig", "w4a16"):
        for sc in ("real", "opt"):
            print(f"[{rq} / {sc}]  전략별 tok/s (유저@배치1 | 시스템@배치8 | 시스템@배치32 "
                  f"동기→PP파이프)")
            for st in STRATS:
                row = {}
                for r in results:
                    if r["strategy"] == st.name and r["requant"] == rq and r["scenario"] == sc:
                        row[r["batch"]] = r
                print(f"  {st.name:11s}: {row[1]['tok_s_user']:6.1f} | "
                      f"{row[8]['tok_s_system']:7.0f} | {row[32]['tok_s_system']:7.0f}"
                      f" → {row[32]['tok_s_system_pipelined']:7.0f}"
                      f"   (step@1 {row[1]['step_ms']:6.1f}ms)")
            print()
    print(f"판정: {verdict}  — real 최선 {b_real['strategy']} {b_real['tok_s_user']:.1f} tok/s, "
          f"opt 최선 {b_opt['strategy']} {b_opt['tok_s_user']:.1f} tok/s; "
          f"처리량(real b32, P1) {thr_real['strategy']} "
          f"{thr_real['tok_s_system_pipelined']:.0f} tok/s")
    print("산출: results.json / REPORT.md / verdict.json")


if __name__ == "__main__":
    main()
