"""Pure-Python DSpark acceptance/state-commit reference for Kimi K3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")
TARGET_AUX_LAYER_IDS = (2, 23, 47, 71, 89)
DSPARK_BLOCK_SIZE = 7


def accepted_state_prefix(states: Sequence[T], accepted: int) -> Sequence[T]:
    """Return exactly the recurrent/conv state transitions that may commit."""
    if not 0 <= accepted <= DSPARK_BLOCK_SIZE:
        raise ValueError(f"accepted must be in [0, 7], got {accepted}")
    return states[:accepted]


def accepted_block_length(
    draft_probabilities: Sequence[float],
    uniforms: Sequence[float],
) -> int:
    """Reference block rejection: stop at the first rejected draft token."""
    if len(draft_probabilities) != len(uniforms):
        raise ValueError("probabilities and uniforms must have equal length")
    if len(draft_probabilities) > DSPARK_BLOCK_SIZE:
        raise ValueError("draft block exceeds DSpark block size 7")
    accepted = 0
    for probability, uniform in zip(draft_probabilities, uniforms):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        if not 0.0 <= uniform <= 1.0:
            raise ValueError("uniforms must be in [0, 1]")
        if uniform > probability:
            break
        accepted += 1
    return accepted


def commit_state_bundle(
    kda_states: Sequence[T],
    conv_states: Sequence[T],
    accepted: int,
) -> tuple[Sequence[T], Sequence[T]]:
    """Apply one acceptance decision identically to KDA and conv state."""
    return (
        accepted_state_prefix(kda_states, accepted),
        accepted_state_prefix(conv_states, accepted),
    )
