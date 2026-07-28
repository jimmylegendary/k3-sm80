"""K3 실측 shape 상수 + S2 SHAPES 게이트 요구 커버리지 + PERF 바이트 모델.

원천(1차): recon/k3-config.md — HF moonshotai/Kimi-K3 @ 9f62e4e9fffbd0a83ddd60e1c209d828994b3569
           (config.json / modeling_kimi_linear.py / model.safetensors.index.json, 2026-07-28 확보)
보조:      recon/kda-fla.md (KDA state 산정·gate_lower_bound), docs/PLAN.md §S2 (배치/ctx 대표점)

규칙: 이 파일의 모든 수치는 recon 문서와 1:1 대조 가능해야 한다(각 상수에 출처 주석).
      새 수치를 추측으로 추가하지 말 것 — recon에 없으면 recon부터 갱신한다.

CLI:  python3 shapes.py --check   → 자기일관성 assert + recon 대조표 출력 (호스트 게이트용)
"""

from __future__ import annotations

import json
import sys
import zlib

# ---------------------------------------------------------------------------
# 1) 모델 전역 (recon/k3-config.md §1)
# ---------------------------------------------------------------------------
HIDDEN = 7168                # §1 hidden_size
LATENT = 3584                # §1 routed_expert_hidden_size (LatentMoE latent dim; README 표 교차확인)
N_EXPERTS = 896              # §1 num_experts
TOPK = 16                    # §1 num_experts_per_token
MOE_INTER = 3072             # §1 moe_intermediate_size (expert FFN 폭 — latent 3584 공간에서 동작)
N_SHARED = 2                 # §1 num_shared_experts (fused 1개 KimiMLP로 구현)
SHARED_FUSED_INTER = 6144    # §1 shared fused intermediate = 2×3072 (hidden 7168 공간, bf16)
DENSE_INTER = 33792          # §1 dense 레이어(레이어 1) intermediate_size
VOCAB = 163840               # §1 vocab_size
N_LAYERS = 93                # §1 num_hidden_layers
KDA_LAYERS = 69              # §1/§2 KDA:MLA = 69:24, [KDA,KDA,KDA,MLA]×23 + [MLA]
MLA_LAYERS = 24
MOE_LAYERS = 92              # §2 FFN: 레이어 1 = dense, 2..93 = MoE
DENSE_LAYERS = 1

# ---------------------------------------------------------------------------
# 2) KDA (recon/k3-config.md §1·§3 KDA층 / recon/kda-fla.md §3)
# ---------------------------------------------------------------------------
KDA_HEADS = 96               # §1 linear_attn_config.num_heads (H=96)
KDA_HEAD_DIM = 128           # §1 head_dim (d_k = d_v = 128)
KDA_PROJ = 12288             # §1 투영폭 = 96×128
KDA_CONV_KERNEL = 4          # §1 short_conv_kernel_size (kda-fla.md §3 K_conv=4)
KDA_F_RANK = 128             # §3 forget gate 저랭크(f_a_proj 7168→128→12288)
KDA_BETA_DIM = 96            # §3 b_proj 7168→96 (head당 beta 스칼라)
KDA_GATE_LOWER_BOUND = -5.0  # kda-fla.md §3 gate_lower_bound (use_full_rank_gate: true)
# KDA state (kda-fla.md §3 — vLLM kda_state_shape/kda_state_dtype 기준):
#   recurrent = (H, d_k, d_v) fp32 하드코딩 → 96×128×128×4 B = 6.29 MB/layer/seq
#   conv      = (3·H·d_k, K_conv−1) = (36864, 3), dtype = model dtype(bf16)
KDA_STATE_BYTES_LAYER_SEQ = KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4   # 6,291,456
KDA_CONV_STATE_ELEMS_LAYER_SEQ = 3 * KDA_PROJ * (KDA_CONV_KERNEL - 1)     # 110,592 (bf16)

# ---------------------------------------------------------------------------
# 3) MLA (recon/k3-config.md §1·§3 Gated MLA층)
# ---------------------------------------------------------------------------
MLA_HEADS = 96               # §1 num_attention_heads (= num_key_value_heads)
MLA_KV_LORA = 512            # §1 kv_lora_rank
MLA_Q_LORA = 1536            # §1 q_lora_rank
MLA_QK_NOPE = 128            # §1 qk_nope_head_dim
MLA_QK_ROPE = 64             # §1 qk_rope_head_dim — 단 mla_use_nope=true → RoPE 미적용(NoPE), 차원은 할당·캐시
MLA_QK_HEAD = 192            # §1 q/k head dim = 128+64
MLA_V_HEAD = 128             # §1 v_head_dim
MLA_CACHE_D = 576            # §8 latent 캐시 = kv_lora 512 + rope몫 64 (vLLM absorbed decode, 2B/elem)
MLA_NOPE = True              # §3 assert self.use_nope, rotary_emb=None

# ---------------------------------------------------------------------------
# 4) 양자화/파라미터 (recon/k3-config.md §3·§4·§6)
# ---------------------------------------------------------------------------
EXPERT_P = 33_030_144        # §3 expert 1개 w1+w2+w3 = 3×3584×3072
BPP_MXFP4 = 0.53125          # §4 MXFP4(E2M1) group32 + E8M0 scale → B/param
BPP_W4A16 = 0.515625         # §6 INT4 g128 + fp16 scale → B/param
SITU_BETA = 4.0              # §1 hidden_act situ (SiTU-GLU; β=4.0)
SITU_LINEAR_BETA = 25.0      # §1 linear-β=25.0

# ---------------------------------------------------------------------------
# 5) S2 테스트 매트릭스 (docs/PLAN.md §S2 완료판정 3: 배치 1/8/32, ctx 8k/128k 대표점)
# ---------------------------------------------------------------------------
BATCHES = (1, 8, 32)
CTX_POINTS = (8192, 131072)

KERNEL_IDS = (
    "k01_kda_chunk",         # KDA chunk (prefill) — FLA 순수 Triton
    "k02_kda_recurrent",     # KDA recurrent (decode)
    "k03_causal_conv1d",     # causal_conv1d (fn=prefill / update=decode)
    "k04_mla_decode",        # MLA decode — TRITON_MLA
    "k05_marlin_moe_w4a16",  # Marlin W4A16 MoE (crash 예상 → S3 입력)
    "k06_dequant_bench",     # dequant 마이크로벤치 (phase0 모델 재보정용)
    "k07_situ_act",          # SiTU 언퓨즈드 참조 (성능 후순위)
)

KERNEL_TITLES = {
    "k01_kda_chunk": "KDA chunk prefill (chunk_kda_with_fused_gate, Triton)",
    "k02_kda_recurrent": "KDA recurrent decode (fused_recurrent_kda, Triton)",
    "k03_causal_conv1d": "causal_conv1d fn/update (depthwise conv4 + SiLU, Triton)",
    "k04_mla_decode": "MLA decode (TRITON_MLA, latent 576 absorbed)",
    "k05_marlin_moe_w4a16": "Marlin W4A16 MoE (marlin_moe_wna16, #35922 대상)",
    "k06_dequant_bench": "W4A16 dequant 마이크로벤치 (dequant_ops/param 재보정)",
    "k07_situ_act": "SiTU-GLU 언퓨즈드 참조 activation",
}

# SHAPES 게이트 요구 커버리지: 케이스 이름 → 케이스 파라미터.
# 하네스는 케이스 이름 목록(covered)을 보고하고, 게이트는 요구 ⊆ covered 로 판정한다.
# decode형: (b, ctx) 5점 / prefill형: (b, T) 4점 — PLAN §S2 "배치 1/8/32; ctx 8k/128k 대표점".
_DECODE_CASES = {
    "b1_ctx8k": dict(b=1, ctx=8192),
    "b8_ctx8k": dict(b=8, ctx=8192),
    "b32_ctx8k": dict(b=32, ctx=8192),
    "b1_ctx128k": dict(b=1, ctx=131072),
    "b32_ctx128k": dict(b=32, ctx=131072),
}
_PREFILL_CASES = {
    "b1_T8k": dict(b=1, T=8192),
    "b1_T128k": dict(b=1, T=131072),
    "b8_T8k_varlen": dict(b=8, T=8192, varlen=True),
    "b32_T8k": dict(b=32, T=8192),
}

REQUIRED_SHAPES = {
    # k01: KDA H=96, dk=dv=128 고정 + prefill 토큰 축 4점 (initial_state 이어받기 1점 포함 권장)
    "k01_kda_chunk": dict(_PREFILL_CASES),
    # k02: decode 5점 (recurrent state는 ctx 무관 고정 크기지만 paged-state 인덱싱 경로 검증 위해 ctx 축 유지)
    "k02_kda_recurrent": dict(_DECODE_CASES),
    # k03: conv kernel=4, 채널 3×12288 — decode update 3점 + prefill fn 2점
    "k03_causal_conv1d": {
        "upd_b1": dict(b=1, mode="update"),
        "upd_b8": dict(b=8, mode="update"),
        "upd_b32": dict(b=32, mode="update"),
        "fn_b1_T8k": dict(b=1, T=8192, mode="fn"),
        "fn_b8_T8k_varlen": dict(b=8, T=8192, mode="fn", varlen=True),
    },
    # k04: MLA decode — kv_lora 512(+rope 64 NoPE) 캐시 576, decode 5점
    "k04_mla_decode": dict(_DECODE_CASES),
    # k05: MoE 896/top16, latent 3584→inter 3072 — 토큰 수 축(m = b×1 decode / prefill 청크)
    "k05_marlin_moe_w4a16": {
        "m1": dict(m=1),
        "m8": dict(m=8),
        "m32": dict(m=32),
        "m8192_prefill": dict(m=8192),
    },
    # k06: expert 단위 W4A16 텐서(3584×3072 g128) + 대형 스트림(활성 16 expert 상당)
    "k06_dequant_bench": {
        "one_expert_w13": dict(K=LATENT, N=MOE_INTER, note="w1/w3 단위"),
        "one_expert_w2": dict(K=MOE_INTER, N=LATENT, note="w2 단위"),
        "active16_experts": dict(params=TOPK * EXPERT_P, note="decode 1토큰 활성분"),
    },
    # k07: SiTU-GLU가 실제로 걸리는 3개 폭 × 토큰 축
    "k07_situ_act": {
        "routed_inter3072_m32": dict(m=32, n=MOE_INTER),
        "shared_inter6144_m32": dict(m=32, n=SHARED_FUSED_INTER),
        "dense_inter33792_m32": dict(m=32, n=DENSE_INTER),
        "shared_inter6144_m8192": dict(m=8192, n=SHARED_FUSED_INTER),
    },
}


def shapes_gate(kernel_id: str, covered: list[str]) -> tuple[str, str, dict]:
    """SHAPES 게이트 판정. covered = 하네스가 실제 실행 완료한 케이스 이름 목록.

    반환: (status, reason, evidence) — status ∈ PASS/FAIL.
    (SKIP은 하네스가 GPU 부재 등으로 케이스를 못 돌린 경우 직접 사유와 함께 설정)
    """
    req = REQUIRED_SHAPES.get(kernel_id)
    if req is None:
        return ("FAIL", f"미등록 커널 id: {kernel_id}", {"known": list(KERNEL_IDS)})
    missing = [c for c in req if c not in covered]
    extra = [c for c in covered if c not in req]
    ev = {
        "required": sorted(req.keys()),
        "covered": sorted(covered),
        "missing": missing,
        "extra": extra,
    }
    if missing:
        return ("FAIL", f"요구 shape 미커버 {len(missing)}건: {', '.join(missing)}", ev)
    return ("PASS", f"요구 {len(req)}케이스 전부 커버", ev)


def stable_seed(kernel_id: str, case_name: str) -> int:
    """(커널, 케이스)별 고정 seed — 재실행·호스트 무관 동일값 (결정성 계약).

    zlib.crc32는 플랫폼/파이썬 버전 무관 안정. 2^31 미만 양수로 접어 torch/np 겸용.
    """
    return zlib.crc32(f"k3-sm80/S2/{kernel_id}/{case_name}".encode()) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# 6) PERF 루프라인 바이트 모델 (커널 단독, 단일 A100 기준)
# ---------------------------------------------------------------------------
def perf_bytes_model(kernel_id: str, **case) -> dict:
    """케이스별 최소 데이터 이동량/연산량 추정 → timing.roofline_us 입력.

    주의: '루프라인 하한'용 추정(입출력 스트림 + 필수 state r/w만 계상).
    커널 내부 재읽기/스크래치는 미포함 — 그래서 PERF 게이트는 eff_floor(임계비율)로 판정.
    산식 출처: recon/k3-config.md §3·§8, recon/kda-fla.md §3 (phase0/k3_model.py 커널항과 동형).
    """
    b = case.get("b", 1)
    T = case.get("T", 0)
    ctx = case.get("ctx", 0)
    m = case.get("m", b)
    out = dict(bytes=0.0, macs=0.0, alu_ops=0.0, note="")

    if kernel_id == "k01_kda_chunk":
        # q,k,v,g 읽기 + o 쓰기 (bf16) + 청크(64)별 inter-chunk state r/w (fp32 h)
        tok = b * T
        io = 5 * tok * KDA_PROJ * 2
        n_chunks = max(1, T // 64) * b
        state = n_chunks * KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4 * 2
        # intra-chunk attention MACs: tok×64×d 두 번(qk^T, ×v) + WY 보정 동차수
        macs = tok * 64 * KDA_HEAD_DIM * KDA_HEADS * 3
        out.update(bytes=io + state, macs=macs,
                   note="io=q/k/v/g/o bf16 스트림, state=64토큰 청크당 h(fp32) r/w")
    elif kernel_id == "k02_kda_recurrent":
        # 지배항: recurrent state (H,dk,dv) fp32 read+write / seq
        state = b * KDA_STATE_BYTES_LAYER_SEQ * 2
        io = b * (4 * KDA_PROJ + KDA_PROJ) * 2   # q,k,v,g 읽기 + o 쓰기
        out.update(bytes=state + io, macs=b * KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 2,
                   note="state 6.29MB/seq fp32 r+w 지배 (kda-fla.md §3)")
    elif kernel_id == "k03_causal_conv1d":
        ch = 3 * KDA_PROJ
        if case.get("mode") == "fn":
            tok = b * T
            out.update(bytes=tok * ch * 2 * 2, macs=tok * ch * KDA_CONV_KERNEL,
                       note="prefill: x 읽기+쓰기 bf16, 4-tap MAC")
        else:
            state = b * KDA_CONV_STATE_ELEMS_LAYER_SEQ * 2 * 2   # bf16 r+w
            io = b * ch * 2 * 2
            out.update(bytes=state + io, macs=b * ch * KDA_CONV_KERNEL,
                       note="decode: conv state(3탭) r/w + x io")
    elif kernel_id == "k04_mla_decode":
        # 지배항: latent KV 캐시 스트림 b×ctx×576×2B (absorbed decode; §8 27,648B/tok/24층 중 1층분)
        kv = b * ctx * MLA_CACHE_D * 2
        q = b * MLA_HEADS * MLA_QK_HEAD * 2
        o = b * MLA_HEADS * MLA_KV_LORA * 2
        macs = b * MLA_HEADS * ctx * (MLA_QK_HEAD + MLA_KV_LORA)
        out.update(bytes=kv + q + o, macs=macs,
                   note="latent MQA: KV 576/tok 스트림 지배 (k3-config §8)")
    elif kernel_id == "k05_marlin_moe_w4a16":
        uniq = min(m * TOPK, N_EXPERTS)
        w = uniq * EXPERT_P * BPP_W4A16
        act = m * TOPK * (2 * LATENT * 2 + 2 * MOE_INTER * 2)
        macs = m * TOPK * 3 * LATENT * MOE_INTER
        out.update(bytes=w + act, macs=macs, alu_ops=uniq * EXPERT_P * 4.0,
                   note="uniq expert 가중치(W4A16 0.515625B/p) + latent 활성; alu=dequant 4ops/p 가정(무근거 밴드 — k06이 재보정)")
    elif kernel_id == "k06_dequant_bench":
        p = case.get("params", case.get("K", LATENT) * case.get("N", MOE_INTER))
        out.update(bytes=p * BPP_W4A16 + p * 2, alu_ops=p * 4.0,
                   note="packed 읽기 + bf16 쓰기; alu는 참고치(측정 대상 그 자체)")
    elif kernel_id == "k07_situ_act":
        n = case["n"]
        out.update(bytes=m * n * 2 * 3, alu_ops=m * n * 12.0,
                   note="gate/up 읽기 + out 쓰기 bf16; SiTU 초월함수 ~12 ops/elem 가정")
    else:
        raise KeyError(f"미등록 커널 id: {kernel_id}")
    return out


# ---------------------------------------------------------------------------
# 7) 자기일관성 + recon 대조표
# ---------------------------------------------------------------------------
_CROSSCHECK = [
    # (상수, 값, recon 출처, recon 기재값)
    ("HIDDEN", HIDDEN, "k3-config §1 hidden_size", 7168),
    ("LATENT", LATENT, "k3-config §1 routed_expert_hidden_size", 3584),
    ("N_EXPERTS", N_EXPERTS, "k3-config §1 num_experts", 896),
    ("TOPK", TOPK, "k3-config §1 num_experts_per_token", 16),
    ("MOE_INTER", MOE_INTER, "k3-config §1 moe_intermediate_size", 3072),
    ("SHARED_FUSED_INTER", SHARED_FUSED_INTER, "k3-config §1 shared fused 2×3072", 6144),
    ("DENSE_INTER", DENSE_INTER, "k3-config §1 dense intermediate", 33792),
    ("VOCAB", VOCAB, "k3-config §1 vocab_size", 163840),
    ("N_LAYERS", N_LAYERS, "k3-config §1 num_hidden_layers", 93),
    ("KDA_LAYERS", KDA_LAYERS, "k3-config §2 KDA 69층", 69),
    ("MLA_LAYERS", MLA_LAYERS, "k3-config §2 MLA 24층", 24),
    ("KDA_HEADS", KDA_HEADS, "k3-config §1 KDA heads", 96),
    ("KDA_HEAD_DIM", KDA_HEAD_DIM, "k3-config §1 dk=dv", 128),
    ("KDA_PROJ", KDA_PROJ, "k3-config §1 투영폭 96×128", 12288),
    ("KDA_CONV_KERNEL", KDA_CONV_KERNEL, "kda-fla §3 short_conv_kernel_size", 4),
    ("MLA_KV_LORA", MLA_KV_LORA, "k3-config §1 kv_lora_rank", 512),
    ("MLA_Q_LORA", MLA_Q_LORA, "k3-config §1 q_lora_rank", 1536),
    ("MLA_QK_NOPE", MLA_QK_NOPE, "k3-config §1 qk_nope_head_dim", 128),
    ("MLA_QK_ROPE", MLA_QK_ROPE, "k3-config §1 qk_rope_head_dim(NoPE)", 64),
    ("MLA_CACHE_D", MLA_CACHE_D, "k3-config §8 512+64", 576),
    ("EXPERT_P", EXPERT_P, "k3-config §3 expert 3×3584×3072", 33_030_144),
    ("KDA_STATE_BYTES_LAYER_SEQ", KDA_STATE_BYTES_LAYER_SEQ, "kda-fla §3 96×128×128×fp32", 6_291_456),
    ("BPP_MXFP4", BPP_MXFP4, "k3-config §6 MXFP4 B/param", 0.53125),
    ("BPP_W4A16", BPP_W4A16, "k3-config §6 INT4 g128 B/param", 0.515625),
]


def self_check(verbose: bool = True) -> bool:
    assert KDA_PROJ == KDA_HEADS * KDA_HEAD_DIM
    assert MLA_QK_HEAD == MLA_QK_NOPE + MLA_QK_ROPE
    assert MLA_CACHE_D == MLA_KV_LORA + MLA_QK_ROPE
    assert SHARED_FUSED_INTER == N_SHARED * MOE_INTER
    assert EXPERT_P == 3 * LATENT * MOE_INTER
    assert KDA_STATE_BYTES_LAYER_SEQ == KDA_HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4
    assert N_LAYERS == KDA_LAYERS + MLA_LAYERS
    assert N_LAYERS == MOE_LAYERS + DENSE_LAYERS
    ok = True
    rows = []
    for name, val, src, expect in _CROSSCHECK:
        good = val == expect
        ok = ok and good
        rows.append((name, val, src, expect, "OK" if good else "MISMATCH"))
    if verbose:
        print(f"{'상수':28s} {'값':>12s}  {'recon 출처':40s} {'대조':>8s}")
        for name, val, src, expect, tag in rows:
            print(f"{name:28s} {val!s:>12s}  {src:40s} {tag:>8s}")
        # PERF 모델 결정성 자기점검 (대표 케이스)
        for kid in KERNEL_IDS:
            case = next(iter(REQUIRED_SHAPES[kid].values()))
            _ = perf_bytes_model(kid, **{k: v for k, v in case.items() if k != "note"})
        print(f"shapes self_check: {'PASS' if ok else 'FAIL'} "
              f"(상수 {len(rows)}건 대조, 커널 {len(KERNEL_IDS)}종 케이스 "
              f"{sum(len(v) for v in REQUIRED_SHAPES.values())}개, seed 예시 "
              f"k01/b1_T8k={stable_seed('k01_kda_chunk', 'b1_T8k')})")
    return ok


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(json.dumps({k: {n: c for n, c in v.items()} for k, v in REQUIRED_SHAPES.items()},
                         ensure_ascii=False, indent=2, sort_keys=True))
        sys.exit(0)
    sys.exit(0 if self_check() else 1)
