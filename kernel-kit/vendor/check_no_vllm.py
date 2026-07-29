#!/usr/bin/env python3
"""vendor/ 호스트 게이트 — AST 수준으로 vLLM 미의존을 증명한다 (torch 불필요).

검사 (전부 stdlib 만 사용 — GPU/torch 없는 저작 호스트·컨테이너 gate 겸용):
  1. py_compile: vendor/ 전 *.py 문법 green.
  2. AST import 스캔: Import/ImportFrom 전수 수집(함수 내부 lazy import 포함) —
     `vllm` 또는 `vllm.*` 이 하나라도 있으면 FAIL.
  3. 외부 의존 allowlist: 상대 import(level>0)·stdlib 를 제외한 절대 import 는
     {torch, triton, numpy, packaging} 만 허용 — 그 밖이 남아 있으면 의존 체인
     vendor 화가 덜 끝난 것이므로 FAIL. (numpy=causal_conv1d, packaging=
     triton_decode_attention 의 원본 의존 — docker triton 티어 이미지에 포함할 것.)

exit: 0 green / 1 위반 존재 / 3 스크립트 오류.
사용: python3 vendor/check_no_vllm.py [--quiet]
"""

from __future__ import annotations

import ast
import os
import py_compile
import sys

VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTERNAL = {"torch", "triton", "numpy", "packaging"}
# 이 스크립트 자신은 스캔 대상에서 제외하지 않는다(자기도 stdlib-only 여야 함).


def stdlib_names() -> frozenset[str]:
    return frozenset(sys.stdlib_module_names)  # py3.10+


def scan_file(path: str) -> tuple[list[str], list[str]]:
    """반환: (vllm 위반 목록, allowlist 밖 외부 import 목록) — 'L<line>: <모듈>' 형식."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    vllm_hits: list[str] = []
    external_bad: list[str] = []
    std = stdlib_names()
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # 상대 import = vendor 내부
            mods = [node.module or ""]
        for m in mods:
            top = m.split(".")[0]
            if top == "vllm":
                vllm_hits.append(f"L{node.lineno}: {m}")
            elif top and top not in std and top not in ALLOWED_EXTERNAL:
                external_bad.append(f"L{node.lineno}: {m}")
    return vllm_hits, external_bad


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    files = []
    for root, _dirs, names in os.walk(VENDOR_DIR):
        if "__pycache__" in root:
            continue
        files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".py")]
    if not files:
        print("[check_no_vllm] vendor/ 에 .py 없음", file=sys.stderr)
        return 3

    n_bad = 0
    for path in sorted(files):
        rel = os.path.relpath(path, VENDOR_DIR)
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            print(f"FAIL {rel}: py_compile — {e.msg.splitlines()[-1] if e.msg else e}")
            n_bad += 1
            continue
        vllm_hits, external_bad = scan_file(path)
        if vllm_hits or external_bad:
            n_bad += 1
            for h in vllm_hits:
                print(f"FAIL {rel}: vllm import 잔존 — {h}")
            for h in external_bad:
                print(f"FAIL {rel}: allowlist 밖 외부 import — {h} "
                      f"(허용: {sorted(ALLOWED_EXTERNAL)})")
        elif not quiet:
            print(f"ok   {rel}")

    if n_bad:
        print(f"[check_no_vllm] FAIL — {n_bad}/{len(files)} 파일 위반", file=sys.stderr)
        return 1
    print(f"[check_no_vllm] green — {len(files)} 파일: py_compile + vllm import 0건 "
          f"+ 외부 의존 ⊆ {sorted(ALLOWED_EXTERNAL)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:  # 스크립트 자체 오류는 game over 가 아니라 3
        print(f"[check_no_vllm] 스크립트 오류: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
