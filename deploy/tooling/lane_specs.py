"""Immutable topology, feature, and model contracts for the serving lanes.

lane 01-05는 Kimi-K3 계열(``family="k3"``), lane 11은 Kimi-K3.1-Pro 계열
(``family="k31"``)이다. 모델 필드의 기본값은 K3 값이므로 lane 01-05의 렌더 결과는
모델 필드를 추가하기 전과 byte-identical이다. 이 불변성은
``tests/deploy/test_render_lanes.py``의 파일별 SHA-256 스냅샷이 고정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

FAMILY_K3 = "k3"
FAMILY_K31 = "k31"

# K3(lane 01-05) 값. LaneSpec 모델 필드의 기본값이라 lane 01-05는 따로 적지 않는다.
K3_TARGET_MODEL = "moonshotai/Kimi-K3"
K3_DRAFT_MODEL = "Inferact/Kimi-K3-DSpark"
K3_IMAGE = "k3-sm80-openai:0498dc7e-sm80-v3"
K3_SERVING_ROOT = "/mnt/nas/home/ubuntu/service/large_model/Kimi-K3"
K3_SERVED_NAME = "kimi-k3"
K3_ATTENTION_BACKEND = "TRITON_MLA"
IMAGE_VERIFIER = "docker/verify_image.py"

# K3.1-Pro(lane 11) 값. vLLM main a3c14efb 핀 위의 SM80 sparse MLA 백엔드
# (PR #47629 계열 TRITON_MLA_SPARSE)와 top-k 백엔드 persistent를 쓴다.
K31_TARGET_MODEL = "moonshotai/Kimi-K3.1-Pro"
K31_IMAGE = "k31-sm80-openai:a3c14efb-sm80-v1"
K31_SERVING_ROOT = "/mnt/nas/home/ubuntu/service/large_model/Kimi-K3.1-Pro"
K31_SERVED_NAME = "kimi-k31-pro"
K31_ATTENTION_BACKEND = "TRITON_MLA_SPARSE"
K31_SPARSE_INDEXER_TOPK_BACKEND = "persistent"


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
    family: str = FAMILY_K3
    target_model: str = K3_TARGET_MODEL
    # 빈 문자열 = draft 없음(DSpark를 쓰지 않는 lane). 키 자체는 .env에 남는다.
    draft_model: str = K3_DRAFT_MODEL
    image: str = K3_IMAGE
    serving_root: str = K3_SERVING_ROOT
    served_name: str = K3_SERVED_NAME
    attention_backend: str = K3_ATTENTION_BACKEND
    # 빈 문자열 = 키를 .env에 쓰지 않는다(K3 lane). 값이 있으면 K3.1 전용 키
    # VLLM_SPARSE_INDEXER_TOPK_BACKEND와 CLI --sparse-indexer-topk-backend가 생긴다.
    sparse_indexer_topk_backend: str = ""
    # preflight가 각 노드에서 실행하는 이미지 검증 스크립트(리포 루트 기준 경로).
    image_verifier: str = IMAGE_VERIFIER

    def __post_init__(self) -> None:
        if self.family not in (FAMILY_K3, FAMILY_K31):
            raise ValueError(f"unknown lane family: {self.family!r}")
        if self.is_k31 and (self.dspark or self.fp8_kv):
            # K3.1 DSpark(draft kv_cache_dtype 주입 등)·software-FP8 KV는 P2/P3이다.
            # 지원하지 않는 조합을 K3 값으로 조용히 렌더하지 않도록 생성 시점에 막는다.
            raise ValueError(
                f"{self.id}: K3.1 lane supports neither dspark nor fp8_kv yet"
            )

    @property
    def number(self) -> int:
        return int(self.id[:2])

    @property
    def is_k31(self) -> bool:
        return self.family == FAMILY_K31

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
    LaneSpec(
        "11-k31-tp8-pp4-256k-safe",
        8,
        4,
        1,
        False,
        262_144,
        False,
        False,
        family=FAMILY_K31,
        target_model=K31_TARGET_MODEL,
        draft_model="",
        image=K31_IMAGE,
        serving_root=K31_SERVING_ROOT,
        served_name=K31_SERVED_NAME,
        attention_backend=K31_ATTENTION_BACKEND,
        sparse_indexer_topk_backend=K31_SPARSE_INDEXER_TOPK_BACKEND,
        image_verifier="docker/verify_k31_image.py",
    ),
)


def lane_by_id(lane_id: str) -> LaneSpec:
    """Resolve a full lane id or its two-digit prefix (``"01"``, ``"11"``)."""

    for lane in LANES:
        if lane.id == lane_id or lane.id.startswith(f"{lane_id}-"):
            return lane
    raise KeyError(f"unknown lane: {lane_id}")


def lane_by_number(number: int, family: str | None = None) -> LaneSpec:
    """Resolve a lane by its numeric prefix, optionally within one family."""

    for lane in LANES:
        if lane.number == number and (family is None or lane.family == family):
            return lane
    raise KeyError(f"unknown lane number: {number} family={family}")


def lanes_in_family(family: str) -> tuple[LaneSpec, ...]:
    return tuple(lane for lane in LANES if lane.family == family)
