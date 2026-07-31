"""Fail-closed evidence stamps bound to code, image, environment, and gate."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StampKey:
    repo_commit: str
    image_digest: str
    env_sha256: str
    gate_id: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def write_stamp(
    path: Path,
    key: StampKey,
    evidence: dict[str, object],
) -> None:
    """Atomically write a stamp. Validation separately enforces PASS evidence."""

    payload: dict[str, object] = {
        "schema_version": 1,
        **dataclasses.asdict(key),
        "evidence": evidence,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_stamp(path: Path, expected: StampKey) -> list[str]:
    if not path.is_file():
        return ["stamp missing"]
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ["stamp is not valid JSON"]
    except OSError as exc:
        return [f"stamp unreadable: {exc}"]
    if not isinstance(actual, dict):
        return ["stamp root is not an object"]

    errors: list[str] = []
    if actual.get("schema_version") != 1:
        errors.append("schema_version mismatch")
    for field in dataclasses.fields(StampKey):
        if actual.get(field.name) != getattr(expected, field.name):
            errors.append(f"{field.name} mismatch")

    evidence = actual.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence missing")
    elif evidence.get("overall") != "PASS":
        errors.append("evidence overall is not PASS")
    return errors
