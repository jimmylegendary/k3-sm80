#!/usr/bin/env python3
"""Lane cluster entrypoint; command construction is wired in Task 3."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("serve", "up", "down", "status", "logs"))
    parser.add_argument("--node-rank", type=int, choices=range(4))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        f"K3-CLUSTER-NOT-WIRED lane=01-tp8-pp4-256k-safe "
        f"action={args.action} node_rank={args.node_rank} dry_run={args.dry_run}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
