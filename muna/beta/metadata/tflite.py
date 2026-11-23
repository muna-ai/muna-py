# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ..compile import CompileTarget
from .litert import LiteRTInterpreterOptions

class TFLiteInterpreterMetadata(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Metadata to compile a TensorFlow Lite `Interpreter` for inference.

    Members:
        interpreter (tensorflow.lite.Interpreter | ai_edge_litert.interpreter.Interpreter): TensorFlow Lite interpreter.
        model_path (str | Path): TFLite model path. The model must exist at this path in the compiler sandbox.
        options (list): TFLite interpreter options.
        targets (list | None): Compile targets where this metadata applies.
    """
    kind: Literal["meta.inference.tflite"] = Field("meta.inference.tflite", init=False)
    interpreter: object = Field(
        description="TensorFlow Lite interpreter to apply metadata to.",
        exclude=True
    )
    model_path: str | Path = Field(
        description="TFLite model path. The model must exist at this path in the compiler sandbox.",
        exclude=True
    )
    options: list[LiteRTInterpreterOptions] | None = Field(
        default=None,
        description="TFLite interpreter options.",
        exclude=True
    )
    targets: list[CompileTarget] | None = Field(
        default=None,
        description="Compile targets where this metadata should apply.",
        exclude=True
    )