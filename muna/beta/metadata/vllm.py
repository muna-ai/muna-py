# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

from ._torch import _validate_torch_module

class vLLMInferenceMetadata(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    """
    Metadata to compile a large language model for multi-GPU inference with vLLM.

    Members:
        model (torch.nn.Module): Large language model to compile.
    """
    kind: Literal["meta.inference.vllm"] = Field(default="meta.inference.vllm", init=False)
    model: Annotated[object, BeforeValidator(_validate_torch_module)] = Field(
        description="Large language model to compile.",
        exclude=True
    )