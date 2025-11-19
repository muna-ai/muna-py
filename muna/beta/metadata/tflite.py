# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

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

class TFLiteInterpreterMetadata(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    """
    Metadata to compile a TensorFlow Lite `Interpreter` for inference.

    Members:
        interpreter (tensorflow.lite.Interpreter | ai_edge_litert.interpreter.Interpreter): TensorFlow Lite interpreter.
        model_path (str | Path): TFLite model path. The model must exist at this path in the compiler sandbox.
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