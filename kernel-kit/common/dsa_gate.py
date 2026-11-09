"""k11/k12 공용 게이트 배선 — child 프로세스 케이스 실행 · compute-sanitizer · 4게이트 조립.

구조(k09 템플릿과 같다): parent 프로세스는 CUDA 를 만지지 않고 케이스마다 child 를 띄운다. child 하나가
메모리 위반으로 죽어도 나머지 케이스가 계속 돈다(MEMORY_VIOLATION 분류). child 는 케이스 하나를 실행해
payload(JSON)를 쓰고, parent 는 payload 들에서 NUMERIC/SHAPES/PERF 를 조립하고 SANITIZER 는 별도로
`compute-sanitizer --tool memcheck` 하에서 소형 케이스를 child 로 실행한다.

child payload 계약 (schema "<kid>-child/v1"):
  {"case", "shape", "status": "PASS"|"FAIL",
   "entries": [{"case": "<name>/<part>", "metrics": {...}, "violations": [...], "passed": bool}, ...],  # NUMERIC 원료
   "timing": {median_us, cv, ...} | null, "extra": {...}}
NUMERIC 게이트는 entries 를 tolerance.NumericResult 로 되살려 tolerance.numeric_gate 에 넘긴다 — 다른 커널과
같은 증거 형식이며, top-k 일치처럼 TolPolicy 로 표현 못 하는 판정도 entry(violations)로 같은 길을 탄다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from common import env_check, report, shapes, timing, tolerance

MEMORY_ERROR = re.compile(r"illegal memory access|invalid (?:__\w+__ )?(?:read|write)|misaligned|out of bounds",
                          flags=re.IGNORECASE)


def kernel_provenance(names: tuple[str, ...]) -> dict:
    """리포트 evidence 용: 하네스가 실제로 실행하는 vendored 커널 파일의 sha256 · 헤더가 주장하는 패치 sha256 · int64 승격 마커 수.

    k11/k12 가 PR 원본이 아니라 이미지에 실리는 패치본을 판정했음을 리포트만으로 대조할 수 있게 한다
    (패치본 파일은 `to(tl.int64)` 마커가 PR 원본보다 많다 — tests/kernel/test_dsa_ops_vendor.py 가 하한을 고정).
    """
    root = Path(__file__).resolve().parents[1] / "vendor" / "dsa_ops"
    out = {}
    for name in names:
        raw = (root / name).read_bytes()
        first = raw.decode("utf-8").splitlines()[0]
        claimed = re.search(r"\(sha256 ([0-9a-f]{64})\)", first)
        out[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                     "header_patch_sha256": claimed.group(1) if claimed else None,
                     "to_int64_markers": raw.decode("utf-8").count("to(tl.int64)")}
    return out


def write_payload(kid: str, payload: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{kid}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / f"{kid}.md").write_text(report.render_md(payload), encoding="utf-8")


def spawn_case(kid: str, harness: Path, case: str, report_dir: Path, timeout_s: int,
               passthrough: list[str] | None = None, launch_blocking: bool = False) -> dict:
    """케이스 하나를 child 프로세스로 실행. 반환: {case,status,classification,returncode,duration_s,result,log_tail}."""
    child_out = report_dir / "children" / f"{case}.json"
    child_out.unlink(missing_ok=True)
    cmd = [sys.executable, str(harness.resolve()), "--child-case", case, "--child-out", str(child_out),
           *(passthrough or [])]
    env = dict(os.environ)
    if launch_blocking:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_s, env=env)
        returncode = proc.returncode
        combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        combined = (exc.stderr if isinstance(exc.stderr, str) else "") + f"\n{kid} child timeout {timeout_s}s"
    try:
        payload = json.loads(child_out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    ok = returncode == 0 and isinstance(payload, dict) and payload.get("status") == "PASS"
    classification = "MEMORY_VIOLATION" if MEMORY_ERROR.search(combined) else ("PASS" if ok else "FAIL")
    return {"case": case, "status": "PASS" if classification == "PASS" else "FAIL",
            "classification": classification, "returncode": returncode,
            "duration_s": round(time.monotonic() - started, 1), "result": payload,
            "log_tail": "" if classification == "PASS" else combined[-3000:]}


def sanitizer_gate(kid: str, harness: Path, sanitizer_cases: list[str], report_dir: Path,
                   timeout_s: int, passthrough: list[str] | None = None) -> tuple[str, str, dict]:
    """소형 케이스 child 를 compute-sanitizer memcheck 하에서 실행 — 전부 ERROR SUMMARY: 0 이어야 PASS."""
    san = env_check.find_compute_sanitizer()
    if not san:
        return "SKIP", "compute-sanitizer 미발견 — full/OpenAI image 에서 실행 필요", {}
    runs = []
    with tempfile.TemporaryDirectory(dir=report_dir) as tmp:
        for case in sanitizer_cases:
            out_json = Path(tmp) / f"{case}.json"
            cmd = [san, "--tool", "memcheck", sys.executable, str(harness.resolve()),
                   "--child-case", case, "--child-out", str(out_json), *(passthrough or [])]
            try:
                proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_s,
                                      env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"})
                rc, combined = proc.returncode, (proc.stderr or "") + "\n" + (proc.stdout or "")
            except subprocess.TimeoutExpired as exc:
                rc, combined = 124, (exc.stderr if isinstance(exc.stderr, str) else "") + "\ncompute-sanitizer timeout"
            # k01~k04 와 같은 파서: 사내 실측(2026-07-30)으로 굳은 분류 — sanitizer 오버헤드 하 allocator 의
            # cudaErrorMemoryAllocation 재시도는 위반이 아니다. (--error-exitcode 는 쓰지 않는다: 그 경우 rc 가
            # 99 로 오염돼 이 정상 케이스도 FAIL 이 된다.)
            st, reason, ev = report.sanitizer_gate_from_output(combined, " ".join(cmd), rc)
            runs.append({"case": case, "status": st, "reason": reason, "returncode": rc,
                         "error_count": ev.get("error_count"), "memory_violations": ev.get("memory_violations"),
                         "api_error_kinds": ev.get("api_error_kinds"), "command": cmd,
                         "sanitizer_lines": [ln for ln in combined.splitlines() if ln.startswith("=========")][-60:]})
    evidence = {"runs": runs}
    for wanted in ("FAIL", "SKIP"):
        hit = next((r for r in runs if r["status"] == wanted), None)
        if hit:
            return wanted, f"[{hit['case']}] {hit['reason']}", evidence
    return "PASS", f"memcheck clean — {len(runs)}개 케이스({', '.join(sanitizer_cases)}): {runs[0]['reason']}", evidence


def numeric_results_from(payloads: list[dict]) -> list[tolerance.NumericResult]:
    out = []
    for p in payloads:
        for e in p.get("entries", []):
            out.append(tolerance.NumericResult(metrics={**e.get("metrics", {}), "case": e.get("case")},
                                               violations=list(e.get("violations", [])),
                                               passed=bool(e.get("passed"))))
    return out


def run_parent(kid: str, harness: Path, pins: dict, case_order: tuple[str, ...], perf_case: str,
               sanitizer_cases: list[str], args: Any, extra_evidence: dict) -> int:
    """GPU 모드 parent: 사전조건 → 케이스 child 순차 실행 → 4게이트 조립 → 리포트 기록."""
    environment = env_check.collect_env()
    ready, checks = env_check.check_preconditions(environment)
    if not ready:
        print(f"[{kid}] GPU 사전조건 미충족", file=sys.stderr)
        for c in checks:
            if c["status"] == "FAIL":
                print(f"  FAIL {c['check']}: {c['detail']}", file=sys.stderr)
        return report.EXIT_PRECOND
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    wanted = [c for c in case_order if not args.cases or c in args.cases]
    passthrough = ["--warmup", str(args.warmup), "--reps", str(args.reps)]
    cases = []
    for name in wanted:
        print(f"{kid}-PARENT case={name} launch", flush=True)
        res = spawn_case(kid, harness, name, report_dir, args.timeout_s, passthrough)
        cases.append(res)
        print(f"{kid}-PARENT case={name} status={res['status']} class={res['classification']} "
              f"duration={res['duration_s']}s", flush=True)

    sanitizer = sanitizer_gate(kid, harness, sanitizer_cases, report_dir, args.timeout_s,
                               passthrough=["--warmup", "0", "--reps", "0"])
    rep = assemble_report(kid, pins, cases, sanitizer, perf_case, args, extra_evidence)
    jp, _ = rep.write(str(report_dir))
    print(f"{kid}-OVERALL {rep.overall()} report={jp}", flush=True)
    return rep.exit_code()


def assemble_report(kid: str, pins: dict, cases: list[dict], sanitizer: tuple[str, str, dict],
                    perf_case: str, args: Any, extra_evidence: dict) -> report.KernelReport:
    """케이스 child 결과 + sanitizer 결과 → 4게이트 리포트 (GPU 없이 단위 시험 가능한 순수 조립 함수)."""
    rep = report.KernelReport(kid, pins=pins)
    for k, v in extra_evidence.items():
        rep.add_evidence(k, v)
    rep.add_evidence("cases", {c["case"]: {k: c[k] for k in ("status", "classification", "returncode",
                                                             "duration_s", "log_tail")} for c in cases})
    executed = [c for c in cases if isinstance(c["result"], dict)]   # 커널이 실제로 돌아 payload 를 남긴 케이스
    rep.add_evidence("case_results", {c["case"]: {k: c["result"].get(k) for k in ("shape", "status", "extra",
                                                                                    "timing")}
                                      for c in executed})
    # --- NUMERIC: child 가 죽거나 payload 가 없는 케이스는 실패 entry 로 포함 (조용히 빠지지 않게) ---
    results = numeric_results_from([c["result"] for c in executed])
    for c in cases:
        if not isinstance(c["result"], dict):
            results.append(tolerance.NumericResult(
                metrics={"case": c["case"], "error": c["classification"]},
                violations=[f"child 실패({c['classification']}, rc={c['returncode']}) — payload 없음"],
                passed=False))
    rep.set_gate("NUMERIC", *tolerance.numeric_gate(kid, results))
    # --- SHAPES: 커널이 실행돼 payload 를 남긴 케이스 = 커버 (오차 판정은 NUMERIC 몫, k01·k04 와 동일 규칙) ---
    rep.set_gate("SHAPES", *shapes.shapes_gate(kid, [c["case"] for c in executed]))
    # --- SANITIZER (소형 케이스, memcheck) ---
    rep.set_gate("SANITIZER", *sanitizer)
    # --- PERF: measure_only 정책 — 대표 케이스 CV 판정, 전 케이스 측정값·루프라인 효율은 evidence ---
    rep.set_gate("PERF", *perf_gate_from(kid, executed, perf_case, args))
    return rep


def perf_gate_from(kid: str, executed: list[dict], perf_case: str, args: Any) -> tuple[str, str, dict]:
    vp = args.phase0_verdict if args.phase0_verdict and os.path.isfile(args.phase0_verdict) else None
    measured = {c["case"]: c["result"].get("timing") for c in executed if c["result"].get("timing")}
    peaks = None
    if vp:
        try:
            tp = timing.resolve_twin_path(vp, args.twin)
            peaks = timing.load_twin_peaks(tp) if tp else None
        except Exception:
            peaks = None
    table = {}
    for name, st in measured.items():
        row = {k: st.get(k) for k in ("median_us", "mean_us", "min_us", "p90_us", "cv", "reps")}
        cfg = shapes.REQUIRED_SHAPES[kid].get(name)
        if peaks and cfg and st.get("median_us"):
            model = shapes.perf_bytes_model(kid, **{k: v for k, v in cfg.items() if k != "note"})
            rl = timing.roofline_us(model["bytes"], model.get("macs", 0.0), model.get("alu_ops", 0.0), peaks)
            row.update(roofline_us=rl["roofline_us"], bound=rl["bound"],
                       roofline_eff=rl["roofline_us"] / st["median_us"],
                       note="효율은 정보용(measure_only — 판정에 쓰지 않음)")
        table[name] = row
    st = measured.get(perf_case)
    cfg = shapes.REQUIRED_SHAPES[kid][perf_case]
    model = shapes.perf_bytes_model(kid, **cfg)
    status, reason, ev = timing.perf_gate(kid, st["median_us"] if st else None, model, vp, args.twin, stats=st)
    ev["perf_case"] = perf_case
    ev["all_cases"] = table
    return status, reason, ev


def gpu_randn_bf16(shape: tuple[int, ...], gen: Any, device: str, chunk_elems: int = 1 << 26,
                   scale: float = 1.0):
    """N(0,1)·scale bf16 텐서를 청크로 생성(fp32 난수 → bf16). gen = 해당 device 의 torch.Generator."""
    import torch
    out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    flat = out.view(-1)
    for i in range(0, flat.numel(), chunk_elems):
        n = min(chunk_elems, flat.numel() - i)
        x = torch.randn(n, dtype=torch.float32, device=device, generator=gen)
        if scale != 1.0:
            x.mul_(scale)
        flat[i:i + n] = x.to(torch.bfloat16)
    return out
