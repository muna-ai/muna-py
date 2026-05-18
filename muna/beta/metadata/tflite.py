# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

from ..compile import CompileTarget

TFLiteInterpreterOptions = Literal[
    # XNNPACK (always on)
    "xnnpack_force_fp16",
    "xnnpack_slow_consistent_arithmetic",
    "xnnpack_subgraph_reshaping",
    # CoreML delegate (iOS / macOS).
    "coreml",
    "coreml_neural_engine_only",
    # GPU delegate (Android).
    "gpu",
    "gpu_min_latency",
    "gpu_sustained_speed",
    "gpu_disallow_precision_loss",
    # NNAPI delegate (Android)
    "nnapi",
    "nnapi_disallow_fp16",
]

def _validate_tflite_interpreter(interpreter: "tensorflow.lite.Interpreter") -> "tensorflow.lite.Interpreter": # type: ignore
    allowed_types = []
    try:
        from tensorflow import lite
        allowed_types.append(lite.Interpreter)
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter
        allowed_types.append(Interpreter)
    except ImportError:
        pass
    if not allowed_types:
        raise ImportError("`tensorflow` or `ai-edge-litert` is required to create this metadata but neither package is installed.")
    if not isinstance(interpreter, tuple(allowed_types)):
        raise ValueError(f"Expected `tensorflow.lite.Interpreter` instance but got `{type(interpreter).__qualname__}`")
    return interpreter

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
    kind: Literal["meta.inference.tflite"] = Field(default="meta.inference.tflite", init=False)
    interpreter: Annotated[object, BeforeValidator(_validate_tflite_interpreter)] = Field(
        description="TensorFlow Lite interpreter to apply metadata to.",
        exclude=True
    )
    model_path: str | Path = Field(
        description="TFLite model path. The model must exist at this path in the compiler sandbox.",
        exclude=True
    )
    options: list[TFLiteInterpreterOptions] | None = Field(
        default=None,
        description="TFLite interpreter options.",
        exclude=True
    )
    targets: list[CompileTarget] | None = Field(
        default=None,
        description="Compile targets where this metadata should apply.",
        exclude=True
    )