#!/usr/bin/env python3
"""vendor/dsa_ops 두 커널 파일을 **이미지에 실리는 패치**에서 재생성·검증하는 개발 도구 (stdlib 전용).

배경: k11/k12 가 판정하는 커널은 컨테이너 이미지에 실제로 실리는 커널이어야 한다. 이미지의 SM80 DSA 커널은 vLLM PR #47629
(head 064801dd, 원저자 haosdent/thomaslwang)를 핀 a3c14efb 에 rebase 한 `docker/patches-k31/triton-mla-sparse-sm80.patch`
의 신규 파일 2개다 — PR 원본 대비 int64 승격(packed hybrid KV row 의 `row*576` 오프셋)과 split-KV 스크래치 workspace
주입(`mid_out_getter` / `max_split_mid_out_numel`)이 들어 있다. vendored 사본 = 이 패치의 신규 파일 본문 + 출처 헤더 3줄 +
`from vllm.* import` 상대화 재배선(§MODMAP)뿐이며, 본문은 무수정이다.

  python3 kernel-kit/vendor/revendor_dsa_ops.py --write                 # 패치 -> vendor/dsa_ops/*.py 재생성
  python3 kernel-kit/vendor/revendor_dsa_ops.py --check                 # 디스크 사본 == 재생성 결과 (exit 0/1)
  python3 kernel-kit/vendor/revendor_dsa_ops.py --check --src-tree DIR  # + 패치 적용 vLLM 트리의 같은 파일과도 본문 대조
  (--patch PATH 로 다른 패치 파일, --dst DIR 로 다른 출력/검사 디렉터리를 지정할 수 있다.
   기본: 리포의 docker/patches-k31/triton-mla-sparse-sm80.patch -> kernel-kit/vendor/dsa_ops/)

패치가 바뀌면(sha256 가 달라지면) 헤더의 sha256 이 낡는다 -> --write 로 재생성하고 STUBS.md §4 의 sha 도 갱신한다.
tests/kernel/test_dsa_ops_vendor.py 가 같은 계약을 독립 구현(역방향 정규화 + 줄 diff)으로 고정한다.

exit: 0 정상 / 1 불일치 / 3 도구 오류.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent
REPO = VENDOR.parents[1]
PATCH_REL = "docker/patches-k31/triton-mla-sparse-sm80.patch"
PIN = "a3c14efbc77c018e8bf579935095a87721349e05"
PR_HEAD = "064801dd2bc6ac2e265dc3fa1f5d803d71bde25d"
# 패치의 신규 파일 경로 -> vendor/dsa_ops 안의 파일명
FILES = {
    "vllm/v1/attention/ops/mqa_logits_triton.py": "mqa_logits_triton.py",
    "vllm/v1/attention/ops/triton_mla_sparse_kernel.py": "triton_mla_sparse_kernel.py",
}
# 재배선: 모듈 경로(전체 일치) -> 상대 import 대상 (dsa_ops 는 vendor/dsa_ops/ 바로 아래: `..` = vendor)
MODMAP = [
    (r"vllm\.triton_utils", ".._vllm_shims.triton_utils"),
    (r"vllm\.utils\.platform_utils", ".._vllm_shims.platform_utils"),
]
FROM_RE = re.compile(r"^(\s*)from ([\w\.]+) import(.*)$")
HEADER_LINES = 3


def patch_sha256(patch: Path) -> str:
    return hashlib.sha256(patch.read_bytes()).hexdigest()


def new_file_bodies(patch_text: str) -> dict[str, str]:
    """`git diff` 의 `new file mode` 구간에서 대상 파일 본문을 복원한다 (`@@ -0,0 +1,N @@` 의 N 줄만 신뢰)."""
    bodies: dict[str, str] = {}
    for section in re.split(r"(?m)^diff --git ", patch_text)[1:]:
        header, _, rest = section.partition("\n")
        match = re.match(r"a/(\S+) b/(\S+)$", header)
        if not match or match.group(2) not in FILES:
            continue
        lines = rest.split("\n")
        if not lines[0].startswith("new file mode"):
            raise ValueError(f"{match.group(2)}: 신규 파일 구간이 아니다 ({lines[0]!r})")
        at = next((i for i, ln in enumerate(lines) if ln.startswith("@@ ")), None)
        hunk = re.match(r"@@ -0,0 \+1,(\d+) @@", lines[at]) if at is not None else None
        if hunk is None:
            raise ValueError(f"{match.group(2)}: `@@ -0,0 +1,N @@` 헝크 머리글 없음")
        body = lines[at + 1:at + 1 + int(hunk.group(1))]
        if len(body) != int(hunk.group(1)) or not all(ln.startswith("+") for ln in body):
            raise ValueError(f"{match.group(2)}: 헝크가 N={hunk.group(1)} 줄 전부 '+' 가 아님")
        bodies[match.group(2)] = "\n".join(ln[1:] for ln in body) + "\n"
    missing = sorted(set(FILES) - set(bodies))
    if missing:
        raise ValueError(f"패치에 대상 파일이 없다: {missing}")
    return bodies


def rewire(source: str, rel: str) -> str:
    """`from vllm.* import …` 줄만 상대 import 로 바꾼다. 매핑 없는 vllm import 가 있으면 실패."""
    out = []
    for line in source.split("\n"):
        match = FROM_RE.match(line)
        if match and (match.group(2) == "vllm" or match.group(2).startswith("vllm.")):
            for pattern, replacement in MODMAP:
                if re.fullmatch(pattern, match.group(2)):
                    line = f"{match.group(1)}from {replacement} import{match.group(3)}"
                    break
            else:
                raise ValueError(f"[{rel}] 매핑 없는 vllm import: {line}")
        out.append(line)
    text = "\n".join(out)
    for node in ast.walk(ast.parse(text)):          # 재배선 뒤 vllm import 0건 증명
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "vllm":
            raise ValueError(f"[{rel}] vllm import 잔존: L{node.lineno} {node.module}")
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "vllm" for a in node.names):
            raise ValueError(f"[{rel}] vllm import 잔존: L{node.lineno}")
    return text


def header(rel: str, sha256: str) -> str:
    return (
        f"# vendored from vllm-project/vllm@{PIN} + k3-sm80 {PATCH_REL} (sha256 {sha256}) {rel} — Apache-2.0\n"
        f"# 원천: PR #47629 head {PR_HEAD} (origin PR #38476) 원저자 haosdent/thomaslwang — 이미지에 실리는 패치본"
        "(PR 원본 + int64 승격 + split-KV 스크래치 workspace 주입 mid_out_getter/max_split_mid_out_numel).\n"
        "# 변경점: 출처 헤더 + vllm.* import 상대화 재배선뿐(예외는 vendor/STUBS.md §4) — 본문 무수정. "
        "라이선스: 아래 원본 SPDX 헤더(Apache-2.0, Copyright contributors to the vLLM project) 그대로 유지.\n"
    )


def render(patch: Path) -> dict[str, str]:
    """vendor/dsa_ops 파일명 -> 파일 전체 내용 (패치에서 결정적으로 생성)."""
    sha = patch_sha256(patch)
    bodies = new_file_bodies(patch.read_text(encoding="utf-8"))
    return {FILES[rel]: header(rel, sha) + rewire(bodies[rel], rel) for rel in FILES}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="vendor/dsa_ops/*.py 를 패치에서 재생성")
    mode.add_argument("--check", action="store_true", help="디스크 사본이 재생성 결과와 같은지 검사")
    ap.add_argument("--patch", type=Path, default=REPO / PATCH_REL)
    ap.add_argument("--dst", type=Path, default=VENDOR / "dsa_ops", help="출력(--write)/검사(--check) 디렉터리")
    ap.add_argument("--src-tree", type=Path, default=None,
                    help="패치가 적용된 vLLM 트리 — 패치 본문과 같은 파일이 트리에도 있는지 대조")
    args = ap.parse_args(argv)
    try:
        rendered = render(args.patch)
        if args.src_tree is not None:
            bodies = new_file_bodies(args.patch.read_text(encoding="utf-8"))
            for rel in FILES:
                tree_text = (args.src_tree / rel).read_text(encoding="utf-8")
                if tree_text != bodies[rel]:
                    print(f"FAIL {rel}: 패치 본문 != --src-tree 파일", file=sys.stderr)
                    return 1
                print(f"ok   {rel}: 패치 본문 == --src-tree 파일")
        bad = 0
        for name, content in rendered.items():
            dst = args.dst / name
            shown = dst.relative_to(REPO) if dst.is_relative_to(REPO) else dst
            if args.write:
                args.dst.mkdir(parents=True, exist_ok=True)
                dst.write_text(content, encoding="utf-8")
                print(f"wrote {shown} ({len(content.splitlines())} lines)")
            elif not dst.is_file() or dst.read_text(encoding="utf-8") != content:
                print(f"FAIL {shown}: 재생성 결과와 다름 (--write 로 갱신)", file=sys.stderr)
                bad += 1
            else:
                print(f"ok   {shown}")
        return 1 if bad else 0
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"[revendor_dsa_ops] 오류: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
