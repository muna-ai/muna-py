# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from textwrap import dedent
from typing import Annotated, Literal, Union

from ._torch import _validate_torch_module

class _SpeculativeDecodingBase(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Common configurations for speculative decoding in large language models.
    """
    draft_model: Annotated[object, BeforeValidator(_validate_torch_module)] = Field(
        description="Drafter module to compile alongside the target model."
    )
    num_draft_tokens: int | None = Field(
        default=None,
        description=dedent("""
        Number of speculative tokens drafted per verify step. `None` 
        means inherit the algorithm's default (DFlash: derive from the 
        drafter config, falling back to 16; EAGLE / EAGLE3: 4). Must 
        be >= 1 when set.
        """).strip(),
        ge=1,
    )

class DFlashSpeculativeDecoding(_SpeculativeDecodingBase):
    """
    DFlash speculative decoding with support for block diffusion draft trees (DDTree).

    DFlash Paper: https://arxiv.org/pdf/2602.06036
    DDTree Paper: https://arxiv.org/pdf/2604.12989

    Members:
        draft_window_size (int | None): Compact draft KV-cache window.
        node_budget (int | None): DDTree node budget. Positive values enable DDTree.
    """
    kind: Literal["dflash"] = Field(default="dflash", init=False)
    draft_window_size: int | None = Field(
        default=None,
        description="Compact draft KV-cache window.",
        ge=1,
    )
    node_budget: int | None = Field(
        default=None,
        description="DDTree node budget. Positive values enable DDTree.",
        ge=0,
    )

class Eagle3SpeculativeDecoding(_SpeculativeDecodingBase):
    """
    EAGLE3 speculative decoding.

    Paper: https://arxiv.org/pdf/2503.01840

    Members:
        num_steps (int): Drafter recurrence depth (tree height).
        topk (int): Top-k sampled per drafter step (tree fan-out).
    """
    kind: Literal["eagle3"] = Field(default="eagle3", init=False)
    num_steps: int = Field(
        default=4,
        description="Drafter recurrence depth (tree height).",
        ge=1,
    )
    topk: int = Field(
        default=8,
        description="Top-k sampled per drafter step (tree fan-out). `1` = chain.",
        ge=1,
    )

SpeculativeDecodingConfig = Annotated[
    Union[
        DFlashSpeculativeDecoding,
        Eagle3SpeculativeDecoding,
    ],
    Field(discriminator="kind"),
]