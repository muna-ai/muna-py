# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

def _validate_ort_inference_session(session: "onnxruntime.InferenceSession") -> "onnxruntime.InferenceSession": # type: ignore
    try:
        from onnxruntime import InferenceSession
        if not isinstance(session, InferenceSession):
            raise ValueError(f"Expected `onnxruntime.InferenceSession` instance but got `{type(session).__qualname__}`")
        return session
    except ImportError:
        raise ImportError("ONNXRuntime is required to create this metadata but it is not installed.")

class OnnxRuntimeInferenceSessionMetadata(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    """
    Metadata to compile an OnnxRuntime `InferenceSession` for inference.

    Members:
        session (onnxruntime.InferenceSession): OnnxRuntime inference session to apply metadata to.
        model_path (str | Path): ONNX model path. The model must exist at this path in the compiler sandbox.
    """
    kind: Literal["meta.inference.onnxruntime"] = Field(default="meta.inference.onnxruntime", init=False)
    session: Annotated[object, BeforeValidator(_validate_ort_inference_session)] = Field(
        description="OnnxRuntime inference session to apply metadata to.",
        exclude=True
    )
    model_path: str | Path = Field(
        description="ONNX model path. The model must exist at this path in the compiler sandbox.",
        exclude=True
    )