# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated

from ._torch import validate_torch_module

class SpeculativeDecodingConfig(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Speculative decoding configuration for large language models.
    """
    draft_model: Annotated[object, BeforeValidator(validate_torch_module)] = Field(
        description="Draft model."
    )
    num_draft_tokens: int | None = Field(
        default=None,
        description="Number of speculative tokens drafted per verify step.",
        ge=1,
    )
    ddtree_node_budget: int | None = Field(
        default=None,
        description="DDTree node budget. Positive values enable DDTree.",
        ge=0,
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