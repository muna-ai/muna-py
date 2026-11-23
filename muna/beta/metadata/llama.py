# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

LlamaCppBackend = Literal["cuda", "vulkan"]

class LlamaCppInferenceMetadata(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Metadata required to lower a Llama.cpp model for LLM inference.
    """
    kind: Literal["meta.inference.llama_cpp"] = Field("meta.inference.llama_cpp", init=False)
    model: object = Field(
        description="Llama model that metadata applies to.",
        exclude=True
    )
    backends: list[LlamaCppBackend] | None = Field(
        default=None,
        description="Llama.cpp hardware backends for accelerated generation.",
        exclude=True
    )