# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ..compile import CompileTarget

OnnxRuntimeExecutionProvider = Literal["cpu", "coreml", "cuda"]
OnnxRuntimeOptimizationLevel = Literal["none", "basic", "extended"]

class OnnxRuntimeInferenceSessionMetadataBase(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    session: object = Field(
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
    targets: list[CompileTarget] | None = Field(
        default=None,
        description="Compile targets where this metadata should apply.",
        exclude=True
    )