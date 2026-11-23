# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, ConfigDict, Field

class SpeculativeDecodingConfig(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Speculative decoding configuration for large language models.
    """
    draft_model: object = Field(description="Draft model.")
    num_draft_tokens: int | None = Field(
        default=None,
        description="Number of speculative tokens drafted per verify step.",
        ge=1,
    )
    eagle3_num_steps: int | None = Field(
        default=None,
        description="EAGLE3 drafter recurrence depth (tree height).",
        ge=1,
    )
    eagle3_topk: int | None = Field(
        default=None,
        description="Top-k sampled per drafter step (tree fan-out). `1` = chain.",
        ge=1,
    )