#!/usr/bin/env python3
"""OpenAI smoke entrypoint; endpoint checks are wired in Task 4."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.parse_args()
    print("K3-SMOKE-NOT-WIRED lane=01-tp8-pp4-256k-safe")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
