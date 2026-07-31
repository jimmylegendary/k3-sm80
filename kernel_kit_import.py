"""Test helper for importing modules from the hyphenated ``kernel-kit`` tree."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


KIT_ROOT = Path(__file__).resolve().parent / "kernel-kit"


def import_common(name: str):
    kit = str(KIT_ROOT)
    if kit not in sys.path:
        sys.path.insert(0, kit)
    return importlib.import_module(f"common.{name}")
