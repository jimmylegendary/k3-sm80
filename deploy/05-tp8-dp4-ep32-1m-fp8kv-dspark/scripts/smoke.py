#!/usr/bin/env python3
"""OpenAI smoke entrypoint; endpoint checks are wired in Task 4."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.parse_args()
    print("K3-SMOKE-NOT-WIRED lane=05-tp8-dp4-ep32-1m-fp8kv-dspark")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
