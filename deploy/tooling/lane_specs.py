"""Immutable topology and feature contracts for the five serving lanes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LaneSpec:
    id: str
    tp: int
    pp: int
    dp: int
    ep: bool
    max_len: int
    dspark: bool
    fp8_kv: bool

    @property
    def number(self) -> int:
        return int(self.id[:2])

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp


LANES: tuple[LaneSpec, ...] = (
    LaneSpec("01-tp8-pp4-256k-safe", 8, 4, 1, False, 262_144, False, False),
    LaneSpec("02-tp8-dp4-ep32-256k-safe", 8, 1, 4, True, 262_144, False, False),
    LaneSpec("03-tp8-dp4-ep32-256k-dspark", 8, 1, 4, True, 262_144, True, False),
    LaneSpec("04-tp8-dp4-ep32-1m-fp8kv", 8, 1, 4, True, 1_048_576, False, True),
    LaneSpec(
        "05-tp8-dp4-ep32-1m-fp8kv-dspark",
        8,
        1,
        4,
        True,
        1_048_576,
        True,
        True,
    ),
)


def lane_by_id(lane_id: str) -> LaneSpec:
    for lane in LANES:
        if lane.id == lane_id or lane.id.startswith(f"{lane_id}-"):
            return lane
    raise KeyError(f"unknown lane: {lane_id}")
