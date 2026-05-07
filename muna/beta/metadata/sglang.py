# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

from ._speculative import SpeculativeDecodingConfig
from ._torch import _validate_torch_module

class SGLangInferenceMetadata(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Metadata to compile a large language model for multi-GPU inference with SGLang.

    Members:
        model (torch.nn.Module): Large language model to compile.
        speculative_decoding (SpeculativeDecodingConfig | None): Speculative decoding configuration.
        max_running_requests (int | None): Maximum concurrent in-flight requests at the scheduler.
    """
    kind: Literal["meta.inference.sglang"] = Field(default="meta.inference.sglang", init=False)
    model: Annotated[object, BeforeValidator(_validate_torch_module)] = Field(
        description="Large language model to compile.",
        exclude=True
    )
    speculative_decoding: SpeculativeDecodingConfig | None = Field(
        default=None,
        description="Speculative decoding configuration.",
        exclude=True
    )
    max_running_requests: int | None = Field(
        default=None,
        description="Maximum concurrent in-flight requests at the scheduler.",
        ge=1,
        exclude=True,
    )