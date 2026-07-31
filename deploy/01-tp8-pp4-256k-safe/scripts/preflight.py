#!/usr/bin/env python3
"""Lane preflight entrypoint; full checks are wired in implementation Task 4."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if not args.env_file.is_file():
        print(f"K3-PREFLIGHT-ENV-MISSING path={args.env_file}")
        return 2
    print("K3-PREFLIGHT-NOT-WIRED lane=01-tp8-pp4-256k-safe")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
