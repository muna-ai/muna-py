# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

OnnxRuntimeExecutionProvider = Literal["cpu", "coreml", "cuda"]
OnnxRuntimeOptimizationLevel = Literal["none", "basic", "extended"]

def _validate_ort_inference_session(session: "onnxruntime.InferenceSession") -> "onnxruntime.InferenceSession": # type: ignore
    try:
        from onnxruntime import InferenceSession
        if not isinstance(session, InferenceSession):
            raise ValueError(f"Expected `onnxruntime.InferenceSession` instance but got `{type(session).__qualname__}`")
        return session
    except ImportError:
        raise ImportError("ONNXRuntime is required to create this metadata but it is not installed.")

class OnnxRuntimeInferenceSessionMetadataBase(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    session: Annotated[object, BeforeValidator(_validate_ort_inference_session)] = Field(
        description="OnnxRuntime inference session to apply metadata to.",
        exclude=True
    )
    model_path: str | Path = Field(
        description="ONNX model path. The model must exist at this path in the compiler sandbox.",
        exclude=True
    )
    external_data_path: str | Path | None = Field(
        default=None,
        description="Path to ONNX external data file (e.g. .onnx.data).",
        exclude=True
    )