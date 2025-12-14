# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

LlamaCppBackend = Literal["cuda", "vulkan"]

def _validate_llama_cpp_model(model: "llama_cpp.llama.Llama") -> "llama_cpp.llama.Llama": # type: ignore
    try:
        from llama_cpp import Llama
        if not isinstance(model, Llama):
            raise ValueError(f"Expected `llama_cpp.llama.Llama` model but got `{type(model).__qualname__}`")
        return model
    except ImportError:
        raise ImportError("`llama-cpp-python` is required to create this metadata but it is not installed.")

class LlamaCppInferenceMetadata(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    """
    Metadata required to lower a Llama.cpp model for LLM inference.
    """
    kind: Literal["meta.inference.llama_cpp"] = "meta.inference.llama_cpp"
    model: Annotated[object, BeforeValidator(_validate_llama_cpp_model)] = Field(
        description="Llama model that metadata applies to.",
        exclude=True
    )
    backends: list[LlamaCppBackend] | None = Field(
        default=None,
        description="Llama.cpp hardware backends for accelerated generation.",
        exclude=True
    )