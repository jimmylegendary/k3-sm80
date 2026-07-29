"""k05 — Marlin W4A16 MoE 단독 하네스 (#35922 crash 재현 + S3 수정 후 회귀 기준).

무엇을 검증하나
  vLLM `fused_marlin_moe`(W4A16, uint4b8 g128)를 vLLM 서빙 스택 없이 단독 호출한다.
  ⚠ **k05 는 vLLM 설치 필수** — Marlin 은 vLLM C-확장(_moe_C)이라 kernel-kit/vendor/
  (torch+triton 티어) 로 대체 불가. venv-k3(kimi-k3 소스빌드) 또는 도커 `:full`
  이미지에서만 실행 (k01~k04·k07 은 vendored 커널로 vLLM 없이 실행 — PLAN §S2).
  참조 = 동일 dequant(bf16) 가중치의 fp32 grouped matmul (양자화 오차는 참조에 포함,
  tolerance.py k05 정책 근거 그대로). 목적의 절반은 #35922 crash 재현이므로:
    - **크기 사다리**: experts 8→64→256→896, hidden(K)축 512→3584→7168 로 crash 경계 탐색.
    - **케이스별 독립 프로세스**: 각 케이스를 자식 프로세스로 실행해 한 crash 가
      전체 런을 죽이지 않게 한다 (rc/시그널/출력 tail 을 crash 증거로 수집).
    - **compute-sanitizer arm**: 소형 clean 케이스 + '최초 crash 케이스(없으면 m1)'를
      memcheck 로 감싸 실행 — ERROR SUMMARY 파싱은 common/report.py 공용 파서.
    - **c_tmp OOB 가설(#36889) 로그 포인트**: ops.cu@d223c900 L708-717 의 할당식
        c_tmp_elems = min(size_n·len(sorted_token_ids), sms·4·moe_block_size·max_thread_n)
                      (moe_block_size==8 이면 ×2)
      를 GEMM1/GEMM2 각각에 대해 커널 호출 **전에** stdout 으로 즉시 출력(K05-LOGPOINT)
      — hard crash 로 프로세스가 죽어도 부모가 tail 에서 회수한다.
      min() 의 앞항(size_n·sorted_len)이 뒤항보다 작으면 batch 의존 과소할당 위험
      (PR #36889 미병합 수정이 제거하려던 바로 그 항) → undersized_risk=True 로 표기.

PASS 기준 (S3 수정 후 회귀 기준으로 설계 — 지금은 FAIL/CRASH 가 예상 결과)
  NUMERIC   : 실행된 모든 비교 케이스(필수 4 + 사다리) tol 내 + crash 0건
  SANITIZER : memcheck ERROR SUMMARY: 0 errors + 비정상 종료 없음
  SHAPES    : 필수 케이스 m1/m8/m32/m8192_prefill (K3 실측 E=896/top16/latent 3584/inter 3072)
              전부 실행-커버
  PERF      : m32 (decode b32, S6 목표축) 측정 ≤ roofline/0.40 (timing.PERF_POLICY)
  → 4게이트 all-PASS 가 S3 완료 판정의 회귀 기준선이다.

업스트림 핀 (전부 raw.githubusercontent.com 에서 커밋 고정 fetch 로 확인, 2026-07-28):
  - vLLM main d223c900: vllm/model_executor/layers/fused_moe/experts/marlin_moe.py
    (fused_marlin_moe 시그니처), tests/kernels/moe/test_moe.py L679-1009 (가중치 준비
    marlin_quantize + 호출 패턴 + atol=4e-2/rtol=0 바),
    csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu L708-717 (c_tmp)·L852-862 (workspace),
    csrc/libtorch_stable/quantization/marlin/marlin.cuh L26/28/31
    (min_thread_n=64, max_thread_n=256, max_par=16).
  - PR #36889 (미병합, 2026-07-14 closed) diff: c_tmp batch 의존 min() 항 제거 +
    kernel<<<...,max_shared_mem>>> → sh_cache_size + cudaFuncSetAttribute 덮어쓰기 수정.
  - kimi-k3 0498dc7e 는 marlin_moe_wna16 를 변경하지 않음 (recon/vllm-upstream.md (5)).

CLI (run_all.sh 계약):
  python3 harness.py --phase0-verdict <verdict.json> --report-dir <dir> [--cpu-dry-run]
                     [--twin <twin.json>] [--cases m1,m32] [--skip-sanitizer]
내부용: --child-case <name> --child-out <json> (케이스 1개 실행), 사다리/필수 공용.
exit: 0 FAIL 없음 / 1 FAIL / 2 GPU 사전조건 미충족 / 3 내부 오류
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

_KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)
from common import env_check, report, shapes, timing, tolerance  # noqa: E402

KID = "k05_marlin_moe_w4a16"
PINS = {"vllm_main": "d223c900", "vllm_kimi_k3": "0498dc7e",
        "hf": "moonshotai/Kimi-K3@9f62e4e9"}

# marlin.cuh@d223c900 L26/28/31 (namespace marlin_moe_wna16) — python 측 미러 상수
MARLIN_MIN_THREAD_N = 64
MARLIN_MAX_THREAD_N = 256
MARLIN_MAX_PAR = 16

# ---------------------------------------------------------------------------
# 케이스 정의 — 필수 4(K3 실측 shape, shapes.REQUIRED_SHAPES 와 이름 일치) + 크기 사다리
# ---------------------------------------------------------------------------
_FULL = dict(E=shapes.N_EXPERTS, K=shapes.LATENT, N=shapes.MOE_INTER, topk=shapes.TOPK)

# (이름, config, 실행 파라미터) — config: E/K/N/topk/m, run: timeout_s/warmup/reps
CASES: dict[str, dict] = {
    # 크기 사다리 (작은 것부터 — crash 경계 탐색; 독립 프로세스라 한 crash 가 다음을 안 막음)
    "L1_e8_k512":    dict(cfg=dict(E=8,   K=512,  N=256,  topk=2,  m=8),
                          timeout=600,  warmup=3, reps=10,
                          note="최소 shape + c_tmp min() 앞항이 뒤항보다 작은 과소할당 위험 구간 프로브"),
    "L2_e64_k1024":  dict(cfg=dict(E=64,  K=1024, N=512,  topk=8,  m=32),
                          timeout=900,  warmup=3, reps=10,
                          note="PR#36889 회귀테스트 규모(e64)와 동급"),
    "L3_e256_k3584": dict(cfg=dict(E=256, K=3584, N=3072, topk=16, m=32),
                          timeout=1800, warmup=3, reps=10,
                          note="expert 수 스케일업, 행렬은 K3 실치수"),
    "L4_e896_k512":  dict(cfg=dict(E=896, K=512,  N=256,  topk=16, m=32),
                          timeout=1500, warmup=3, reps=10,
                          note="expert=896 축 단독 분리 (행렬은 소형)"),
    # 필수 4 케이스 (shapes.REQUIRED_SHAPES[k05] 와 이름 일치, K3 실측 shape)
    "m1":             dict(cfg=dict(m=1,    **_FULL), timeout=2700, warmup=10, reps=50,
                           note="decode 단일 스트림 — moe_block_size=8 (c_tmp ×2 분기)"),
    "m8":             dict(cfg=dict(m=8,    **_FULL), timeout=2700, warmup=10, reps=50,
                           note="decode b8"),
    "m32":            dict(cfg=dict(m=32,   **_FULL), timeout=2700, warmup=10, reps=50,
                           note="decode b32 — PERF 게이트 판정 케이스"),
    "m8192_prefill":  dict(cfg=dict(m=8192, **_FULL), timeout=5400, warmup=2,  reps=8,
                           note="prefill 청크 — #35922 원보고가 profile_run(대형 m)에서 crash"),
    # hidden 축 상한 스트레스 (K3 expert GEMM 은 latent 3584 이지만 사다리 상한 7168 확인)
    "L6_e896_k7168": dict(cfg=dict(E=896, K=7168, N=3072, topk=16, m=32),
                          timeout=3600, warmup=3, reps=10,
                          note="hidden 축 상한(7168) — K3 hidden 직결 폭, 경계 탐색 최종점"),
}
REQUIRED = list(shapes.REQUIRED_SHAPES[KID].keys())          # m1, m8, m32, m8192_prefill
ORDER = ["L1_e8_k512", "L2_e64_k1024", "L3_e256_k3584", "L4_e896_k512",
         "m1", "m8", "m32", "m8192_prefill", "L6_e896_k7168"]
PERF_CASE = "m32"


# ---------------------------------------------------------------------------
# c_tmp/workspace 로그 포인트 — ops.cu@d223c900 의 python 미러 (#36889 검증용)
# ---------------------------------------------------------------------------
def select_block_size_m(m: int, topk: int, e: int) -> int:
    """experts/marlin_moe.py@d223c900 L324-326 미러 (input_dtype=None → 8 하한 유지)."""
    for block_size_m in [8, 16, 32, 48, 64]:
        if m * topk / e / block_size_m < 0.9:
            break
    return block_size_m


def ctmp_logpoints(cfg: dict, sms: int, sorted_len: int | None = None) -> dict:
    """GEMM1(w13: size_n=2N,size_k=K) / GEMM2(w2: size_n=K,size_k=N) c_tmp·workspace 산식.

    sorted_len 미지정 시 해석적 상한(topk_ids.numel() + E·(bs−1))을 쓰고 estimate 로 표기.
    ops.cu L708-717: max_c_tmp_size = min(size_n·sorted_len, sms·4·bs·max_thread_n) [·2 if bs==8]
    ops.cu L852-862: min_workspace = min((size_n/min_thread_n)·(sorted_len/bs), sms·4)
    """
    m, e, topk = cfg["m"], cfg["E"], cfg["topk"]
    bs = select_block_size_m(m, topk, e)
    est = sorted_len is None
    if est:
        sorted_len = m * topk + e * (bs - 1)   # moe_align 패딩 상한 (케이스별 실값은 자식이 갱신)
    gems = {}
    for tag, size_n, size_k, size_m in (("gemm1_w13", 2 * cfg["N"], cfg["K"], m),
                                        ("gemm2_w2", cfg["K"], cfg["N"], m * topk)):
        arm1 = size_n * sorted_len
        arm2 = sms * 4 * bs * MARLIN_MAX_THREAD_N
        elems = min(arm1, arm2) * (2 if bs == 8 else 1)
        n_tiles = size_n // MARLIN_MIN_THREAD_N
        gems[tag] = {
            "size_m": size_m, "size_n": size_n, "size_k": size_k,
            "arm1_sizen_x_sortedlen": arm1, "arm2_sms4_bs_maxthreadn": arm2,
            "c_tmp_elems": elems, "c_tmp_bytes_fp32": elems * 4,
            "governed_by": "arm1(batch의존)" if arm1 < arm2 else "arm2(sms캡)",
            "undersized_risk_pr36889": arm1 < arm2,
            "n_tiles": n_tiles,
            "min_workspace": min(n_tiles * (sorted_len // bs), sms * 4),
        }
    return {"block_size_m": bs, "sorted_len": sorted_len,
            "sorted_len_is_estimate": est, "sms": sms,
            "constants": {"min_thread_n": MARLIN_MIN_THREAD_N,
                          "max_thread_n": MARLIN_MAX_THREAD_N, "max_par": MARLIN_MAX_PAR},
            "gemms": gems,
            "basis": "ops.cu@d223c900 L708-717/L852-862 미러; PR#36889(미병합)는 arm1 제거"}


# ---------------------------------------------------------------------------
# 자식 프로세스: 케이스 1개 실행 (GPU)
# ---------------------------------------------------------------------------
def _emit(tag: str, obj) -> None:
    """crash 대비 — 판정 전 정보를 즉시 stdout 으로 (부모가 tail 에서 회수)."""
    print(f"{tag} {json.dumps(obj, ensure_ascii=False)}", flush=True)


def run_child_case(case: str, out_path: str) -> int:
    import torch
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )
    from vllm.scalar_type import scalar_types
    try:
        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
            fused_marlin_moe,
        )
        fmm_path = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
    except ImportError:   # 구경로 fallback (브랜치 드리프트 대비)
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (  # type: ignore
            fused_marlin_moe,
        )
        fmm_path = "vllm.model_executor.layers.fused_moe.fused_marlin_moe"

    spec = CASES[case]
    cfg = spec["cfg"]
    E, K, N, topk, m = cfg["E"], cfg["K"], cfg["N"], cfg["topk"], cfg["m"]
    group_size = 128                      # W4A16 g128 (recon/k3-config.md §6 시나리오)
    qtype = scalar_types.uint4b8          # GPTQ형 대칭 INT4 (compressed-tensors W4A16 대응)
    dev = torch.device("cuda:0")
    seed = shapes.stable_seed(KID, case)
    torch.manual_seed(seed)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    props = torch.cuda.get_device_properties(0)
    _emit("K05-CASE", {"case": case, "cfg": cfg, "seed": seed, "sms": sms,
                       "shared_mem_per_block_optin":
                           getattr(props, "shared_memory_per_block_optin", None),
                       "fused_marlin_moe": fmm_path})

    # 1) 라우팅 먼저 고정 (참조 기여 계산을 가중치 생성 루프에 합치기 위함)
    g_route = torch.Generator().manual_seed(seed)
    score = torch.randn((m, E), generator=g_route, dtype=torch.float32)
    score = torch.softmax(score, dim=-1)
    topk_w_cpu, topk_ids_cpu = torch.topk(score, topk)
    a_cpu = (torch.randn((m, K), generator=g_route, dtype=torch.float32) / 10)
    a = a_cpu.to(dev, torch.bfloat16)
    a_f32 = a.float()                       # 참조 입력 = 커널 입력과 동일 bf16 값
    topk_weights = topk_w_cpu.to(dev).contiguous()            # fp32 (커널 요구)
    topk_ids = topk_ids_cpu.to(dev, torch.int32).contiguous()

    # 2) c_tmp 로그 포인트 — moe_align 실값으로 산출해 커널 호출 전에 출력
    bs = select_block_size_m(m, topk, E)
    sorted_token_ids, expert_ids, num_post_pad = moe_align_block_size(
        topk_ids, bs, E, None, ignore_invalid_experts=True)
    lp = ctmp_logpoints(cfg, sms, sorted_len=int(sorted_token_ids.numel()))
    _emit("K05-LOGPOINT", {"case": case, **lp})

    # 3) expert 별 가중치 생성→양자화→(라우팅된 expert 면) 참조 기여 누적 — 단일 패스.
    #    E=896 전 expert 의 bf16 원본/w_ref 를 동시 상주시키지 않는다 (VRAM 절약).
    qw1_l, s1_l, g1_l, p1_l = [], [], [], []
    qw2_l, s2_l, g2_l, p2_l = [], [], [], []
    ref_out = torch.zeros((m, K), dtype=torch.float32, device=dev)
    routed: dict[int, list[tuple[int, int]]] = {}
    for t in range(m):
        for s in range(topk):
            routed.setdefault(int(topk_ids_cpu[t, s]), []).append((t, s))
    t_build0 = time.time()
    for e_i in range(E):
        g_w = torch.Generator().manual_seed(seed + 1000 + e_i)
        w13 = (torch.randn((2 * N, K), generator=g_w, dtype=torch.float32) / 10) \
            .to(torch.bfloat16)
        w2 = (torch.randn((K, N), generator=g_w, dtype=torch.float32) / 10) \
            .to(torch.bfloat16)
        # test_moe.py L846-859 패턴: marlin_quantize(w.T) → w_ref,(K,2N)/(N,K)
        w_ref13, qw, sc, gi, si, _ = marlin_quantize(
            w13.to(dev).transpose(0, 1).contiguous(), qtype, group_size, act_order=False)
        qw1_l.append(qw); s1_l.append(sc); g1_l.append(gi); p1_l.append(si)
        w_ref2, qw, sc, gi, si, _ = marlin_quantize(
            w2.to(dev).transpose(0, 1).contiguous(), qtype, group_size, act_order=False)
        qw2_l.append(qw); s2_l.append(sc); g2_l.append(gi); p2_l.append(si)
        if e_i in routed:   # fp32 참조 기여 (dequant 가중치 fp32 grouped matmul)
            idx = routed[e_i]
            tks = torch.tensor([t for t, _ in idx], device=dev)
            x = a_f32[tks]                                    # (t_e, K)
            h = x @ w_ref13.float()                           # (t_e, 2N)
            act = torch.nn.functional.silu(h[:, :N]) * h[:, N:]
            y = act @ w_ref2.float()                          # (t_e, K)
            wgt = torch.tensor([float(topk_w_cpu[t, s]) for t, s in idx],
                               device=dev).unsqueeze(1)
            ref_out.index_add_(0, tks, y * wgt)
        del w13, w2, w_ref13, w_ref2
    qw1 = torch.stack(qw1_l).contiguous(); s1 = torch.stack(s1_l)
    qi1 = torch.stack(g1_l); sp1 = torch.stack(p1_l)
    qw2 = torch.stack(qw2_l).contiguous(); s2 = torch.stack(s2_l)
    qi2 = torch.stack(g2_l); sp2 = torch.stack(p2_l)
    del qw1_l, s1_l, g1_l, p1_l, qw2_l, s2_l, g2_l, p2_l
    torch.cuda.synchronize()
    _emit("K05-BUILD", {"case": case, "build_s": round(time.time() - t_build0, 1),
                        "vram_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2)})

    def kernel():
        return fused_marlin_moe(
            a, qw1, qw2, None, None, s1, s2, topk_weights, topk_ids,
            quant_type_id=qtype.id, global_num_experts=E, expert_map=None,
            g_idx1=qi1, g_idx2=qi2, sort_indices1=sp1, sort_indices2=sp2,
            is_k_full=True)

    # 4) 본 실행 (여기가 #35922 crash 지점 후보) + 수치 비교
    out = kernel()
    torch.cuda.synchronize()
    nr = tolerance.compare(KID, ref_out, out, case=case)
    _emit("K05-NUMERIC", {"case": case, "passed": nr.passed,
                          "violations": nr.violations,
                          "metrics": {k: v for k, v in nr.metrics.items()
                                      if k not in ("policy",)}})

    # 5) 타이밍 (판정은 PERF_CASE 만, 나머지는 evidence)
    st = timing.measure(kernel, warmup=spec["warmup"], reps=spec["reps"])

    res = {"case": case, "cfg": cfg, "seed": seed, "logpoints": lp,
           "numeric": {"metrics": nr.metrics, "violations": nr.violations,
                       "passed": nr.passed},
           "timing": st, "fused_marlin_moe": fmm_path,
           "vram_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=True)
    _emit("K05-DONE", {"case": case, "median_us": st["median_us"]})
    return 0


# ---------------------------------------------------------------------------
# 부모: 케이스 subprocess 실행/수집
# ---------------------------------------------------------------------------
def _classify_rc(rc: int) -> str:
    if rc == 0:
        return "ok"
    if rc == 4:
        return "python_exception"      # CUDA IMA 가 RuntimeError 로 떠도 여기 (메시지 참조)
    if rc < 0:
        try:
            return f"signal_{signal.Signals(-rc).name}"
        except ValueError:
            return f"signal_{-rc}"
    if rc in (134, 139):
        return {134: "abort_SIGABRT", 139: "segfault_SIGSEGV"}[rc]
    return f"exit_{rc}"


def spawn_case(case: str, args, extra_env: dict | None = None,
               sanitizer: str | None = None) -> dict:
    spec = CASES[case]
    out_json = os.path.join(args.report_dir, "k05_cases",
                            f"{case}{'.san' if sanitizer else ''}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    if os.path.exists(out_json):
        os.remove(out_json)
    cmd = [sys.executable, os.path.abspath(__file__),
           "--child-case", case, "--child-out", out_json]
    if sanitizer:
        cmd = [sanitizer, "--tool", "memcheck", "--error-exitcode", "88"] + cmd
    env = dict(os.environ)
    env.update(extra_env or {})
    t0 = time.time()
    timeout = spec["timeout"] * (3 if sanitizer else 1)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        rc, out_txt = p.returncode, (p.stdout or "") + (p.stderr or "")
        timed_out = False
    except subprocess.TimeoutExpired as te:
        rc, timed_out = 124, True
        out_txt = ((te.stdout or b"").decode(errors="replace") if isinstance(te.stdout, bytes)
                   else (te.stdout or "")) + \
                  ((te.stderr or b"").decode(errors="replace") if isinstance(te.stderr, bytes)
                   else (te.stderr or ""))
    dur = round(time.time() - t0, 1)
    result = None
    if os.path.exists(out_json):
        try:
            with open(out_json, encoding="utf-8") as f:
                result = json.load(f)
        except Exception:
            result = None
    # crash 여도 stdout 에 찍힌 로그 포인트를 회수
    lp_lines = [ln for ln in out_txt.splitlines()
                if ln.startswith(("K05-LOGPOINT", "K05-CASE", "K05-NUMERIC", "K05-ERROR"))]
    rec = {"case": case, "rc": rc, "class": ("timeout" if timed_out else _classify_rc(rc)),
           "duration_s": dur, "sanitizer": bool(sanitizer),
           "result_json": out_json if result else None,
           "stdout_markers": lp_lines[-8:], "tail": out_txt[-1800:]}
    if result:
        rec["result"] = result
    return rec


def sanitizer_arm(args, first_crash: str | None) -> tuple[str, str, dict]:
    """소형 clean 케이스 + (최초 crash 케이스 또는 m1) 을 memcheck 로 실행."""
    san = env_check.find_compute_sanitizer()
    if not san:
        return ("SKIP", "compute-sanitizer 미발견 — RUNNER.md §6 (CUDA 툴킷 bin PATH)", {})
    targets = ["L1_e8_k512", first_crash or "m1"]
    targets = list(dict.fromkeys(targets))       # 중복 제거, 순서 유지
    runs, statuses = [], []
    for c in targets:
        print(f"[k05] sanitizer arm: {c} (memcheck — 느림, hang 아님)")
        rec = spawn_case(c, args, sanitizer=san)
        st, rs, ev = report.sanitizer_gate_from_output(
            rec["tail"], cmd=f"compute-sanitizer memcheck {c}",
            returncode=rec["rc"] if rec["class"] != "timeout" else None)
        if rec["class"] == "timeout":
            st, rs = "SKIP", f"sanitizer 하 {c} timeout({rec['duration_s']}s) — 판독 불가, 재시도 필요"
        runs.append({"case": c, "status": st, "reason": rs, "rc": rec["rc"],
                     "class": rec["class"], "duration_s": rec["duration_s"],
                     "evidence": ev})
        statuses.append(st)
    ev = {"sanitizer_bin": san, "runs": runs}
    if "FAIL" in statuses:
        bad = next(r for r in runs if r["status"] == "FAIL")
        return ("FAIL", f"[{bad['case']}] {bad['reason']}", ev)
    if all(s == "PASS" for s in statuses):
        return ("PASS", f"memcheck clean: {', '.join(r['case'] for r in runs)}", ev)
    return ("SKIP", "; ".join(f"[{r['case']}] {r['reason']}" for r in runs
                              if r["status"] == "SKIP"), ev)


# ---------------------------------------------------------------------------
# dry-run: 순수 python 참조 MoE 미니 검증 + 케이스 정의/로그포인트 검증 (torch 불필요)
# ---------------------------------------------------------------------------
def _dry_mini_moe() -> dict:
    """E=4/K=8/N=4/topk=2/m=3 순수 python MoE — 참조 로직 자기검증(결정적)."""
    import random
    rnd = random.Random(shapes.stable_seed(KID, "dry_mini"))
    E, K, N, topk, m = 4, 8, 4, 2, 3

    def mat(r, c):
        return [[rnd.gauss(0, 0.1) for _ in range(c)] for _ in range(r)]

    def matmul(A, B):
        return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]

    def silu(v):
        return v / (1.0 + math.exp(-v))

    w13 = [mat(K, 2 * N) for _ in range(E)]
    w2 = [mat(N, K) for _ in range(E)]
    x = mat(m, K)
    ids = [sorted(rnd.sample(range(E), topk)) for _ in range(m)]
    wgt = [[1.0 / topk] * topk for _ in range(m)]

    def forward():
        out = [[0.0] * K for _ in range(m)]
        for t in range(m):
            for s, e in enumerate(ids[t]):
                h = matmul([x[t]], w13[e])[0]
                act = [silu(h[j]) * h[N + j] for j in range(N)]
                y = matmul([act], w2[e])[0]
                for j in range(K):
                    out[t][j] += wgt[t][s] * y[j]
        return [v for row in out for v in row]

    ref = forward()
    again = forward()                       # 결정성: 동일 입력 → 동일 출력
    nr = tolerance.compare(KID, ref, again, case="dry_mini_selfconsistency")
    pert = [v * (1 + 0.2) for v in ref]     # FAIL 방향도 판정기가 잡는지
    nr_bad = tolerance.compare(KID, ref, pert, case="dry_mini_perturbed")
    return {"selfconsistency_passed": nr.passed and nr.metrics["max_abs"] == 0.0,
            "perturbed_detected": not nr_bad.passed,
            "elems": len(ref)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase0-verdict", default=None)
    ap.add_argument("--report-dir", default=os.path.join(_KIT, "kernel-report"))
    ap.add_argument("--twin", default=None)
    ap.add_argument("--cpu-dry-run", action="store_true")
    ap.add_argument("--cases", default=None, help="쉼표구분 케이스 부분 실행(기본 전체)")
    ap.add_argument("--skip-sanitizer", action="store_true")
    ap.add_argument("--child-case", default=None)
    ap.add_argument("--child-out", default=None)
    args = ap.parse_args()

    # ---- 자식 모드 -------------------------------------------------------
    if args.child_case:
        try:
            return run_child_case(args.child_case, args.child_out)
        except Exception as e:  # CUDA IMA RuntimeError 포함 — 증거 남기고 rc 4
            _emit("K05-ERROR", {"case": args.child_case,
                                "error": f"{type(e).__name__}: {e}"[:800]})
            if args.child_out:
                with open(args.child_out + ".err", "w", encoding="utf-8") as f:
                    import traceback
                    f.write(traceback.format_exc())
            return 4

    # ---- 부모 모드 -------------------------------------------------------
    rep = report.KernelReport(KID, dry_run=args.cpu_dry_run, pins=PINS)
    rep.add_evidence("upstream", {
        "issue": "vllm-project/vllm#35922 (stale-close, 미수정)",
        "fix_candidate": "PR#36889 미병합 — c_tmp batch의존 min()항 제거 + "
                         "sh_cache_size 런치 + cudaFuncSetAttribute 덮어쓰기 수정",
        "kernel_src": "csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu@d223c900 "
                      "L708-717(c_tmp) L852-862(workspace)",
        "call_pattern": "tests/kernels/moe/test_moe.py@d223c900 L982-1007",
        "s3_reuse": "이 하네스의 4게이트 all-PASS 가 S3 수정 완료의 회귀 기준선"})

    env = env_check.collect_env()
    ok, checks = env_check.check_preconditions(env, cpu_dry_run=args.cpu_dry_run)
    if not ok:
        print("[k05] GPU 사전조건 미충족 — skip (env_check 상세 참조)", file=sys.stderr)
        for c in checks:
            print(f"  {c['status']:4s} {c['check']:20s} {c['detail']}", file=sys.stderr)
        return report.EXIT_PRECOND

    sel = ORDER if not args.cases else [c for c in ORDER if c in args.cases.split(",")]

    if args.cpu_dry_run:
        # 케이스 정의 검증 + 로그포인트 산식(해석적) + 순수 python 참조 자기검증
        lps = {c: ctmp_logpoints(CASES[c]["cfg"], sms=108) for c in sel}
        mini = _dry_mini_moe()
        rep.add_evidence("dry_run", {
            "logpoints_analytic_sms108": lps, "mini_reference": mini,
            "cases_validated": sel})
        risk = [c for c in sel
                if any(g["undersized_risk_pr36889"] for g in lps[c]["gemms"].values())]
        rep.add_evidence("ctmp_undersized_risk_cases", risk)
        ok_mini = mini["selfconsistency_passed"] and mini["perturbed_detected"]
        rep.set_gate("NUMERIC", "SKIP" if ok_mini else "FAIL",
                     "cpu-dry-run — 순수 python 참조 자기검증 green, 실커널 비교 없음"
                     if ok_mini else "dry-run 참조 자기검증 실패 — 하네스 결함",
                     {"mini": mini})
        rep.set_gate("SANITIZER", "SKIP", "cpu-dry-run — GPU 미실행")
        rep.set_gate("SHAPES", "SKIP",
                     f"cpu-dry-run — 케이스 정의 {len(sel)}건 검증만(필수 {len(REQUIRED)} 포함), "
                     "실행 커버 아님",
                     {"validated": sel, "required": REQUIRED})
        rep.set_gate("PERF", "SKIP", "cpu-dry-run — 측정 없음")
        j, m_ = rep.write(args.report_dir)
        print(f"[k05] dry-run 리포트: {j} (overall={rep.overall()})")
        return rep.exit_code()

    # ---- GPU 본 실행 -----------------------------------------------------
    records: list[dict] = []
    first_crash: str | None = None
    for c in sel:
        print(f"[k05] ── {c} ({CASES[c]['cfg']}) — 독립 프로세스")
        rec = spawn_case(c, args)
        records.append(rec)
        print(f"[k05] {c}: {rec['class']} ({rec['duration_s']}s)")
        if rec["class"] != "ok" and first_crash is None:
            first_crash = c

    # NUMERIC — 실행된 모든 비교 + crash 는 실패 케이스로 산입 (S3 회귀 기준)
    nrs: list[tolerance.NumericResult] = []
    for rec in records:
        if rec.get("result"):
            n = rec["result"]["numeric"]
            nrs.append(tolerance.NumericResult(
                metrics=n["metrics"], violations=n["violations"], passed=n["passed"]))
        else:
            nrs.append(tolerance.NumericResult(
                metrics={"case": rec["case"], "n": 0},
                violations=[f"프로세스 {rec['class']} (rc={rec['rc']}) — 출력 없음"],
                passed=False))
    st, rs, ev = tolerance.numeric_gate(KID, nrs)
    rep.set_gate("NUMERIC", st, rs, ev)

    # SHAPES — 필수 케이스 중 결과가 나온 것만 커버로 인정 (crash → 미커버 FAIL)
    covered = [r["case"] for r in records if r.get("result")]
    st, rs, ev = shapes.shapes_gate(KID, covered)
    rep.set_gate("SHAPES", st, rs, ev)

    # SANITIZER arm
    if args.skip_sanitizer:
        rep.set_gate("SANITIZER", "SKIP", "--skip-sanitizer 지정 (부분 재실행 모드)")
    else:
        st, rs, ev = sanitizer_arm(args, first_crash)
        rep.set_gate("SANITIZER", st, rs, ev)

    # PERF — m32 판정, 나머지는 evidence
    perf_rec = next((r for r in records if r["case"] == PERF_CASE and r.get("result")), None)
    model = shapes.perf_bytes_model(KID, **CASES[PERF_CASE]["cfg"])
    measured = perf_rec["result"]["timing"]["median_us"] if perf_rec else None
    stats = perf_rec["result"]["timing"] if perf_rec else None
    st, rs, ev = timing.perf_gate(KID, measured, model, args.phase0_verdict,
                                  args.twin, stats=stats)
    if perf_rec is None and st == "SKIP":
        rs = f"PERF 판정 케이스({PERF_CASE}) 미완주 — {rs}"
    rep.set_gate("PERF", st, rs, ev)

    # 사다리/crash 증거
    rep.add_evidence("ladder", [
        {"case": r["case"], "cfg": CASES[r["case"]]["cfg"], "class": r["class"],
         "rc": r["rc"], "duration_s": r["duration_s"],
         "median_us": (r.get("result") or {}).get("timing", {}).get("median_us"),
         "numeric_passed": (r.get("result") or {}).get("numeric", {}).get("passed"),
         "logpoints": (r.get("result") or {}).get("logpoints")
         or _scrape_logpoint(r), }
        for r in records])
    rep.add_evidence("first_crash_case", first_crash)
    rep.add_evidence("crash_tails", {r["case"]: r["tail"] for r in records
                                     if r["class"] != "ok"})

    j, m_ = rep.write(args.report_dir)
    print(f"[k05] 리포트: {j} (overall={rep.overall()}"
          f"{', first_crash=' + first_crash if first_crash else ''})")
    return rep.exit_code()


def _scrape_logpoint(rec: dict) -> dict | None:
    """crash 케이스 — stdout 마커에서 K05-LOGPOINT 회수."""
    for ln in rec.get("stdout_markers", []):
        if ln.startswith("K05-LOGPOINT "):
            try:
                return json.loads(ln[len("K05-LOGPOINT "):])
            except Exception:
                return None
    return None


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(report.EXIT_ERR)
