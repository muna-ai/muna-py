# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pathlib import Path
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

from ._torch import PyTorchInferenceMetadataBase

def _validate_litert_interpreter(interpreter: "tensorflow.lite.Interpreter") -> "tensorflow.lite.Interpreter": # type: ignore
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

class LiteRTInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference with LiteRT.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        model_args (tuple[Tensor,...]): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        output_keys (list): Model output dictionary keys. Use this if the model returns a dictionary.
    """
    kind: Literal["meta.inference.litert"] = Field(default="meta.inference.litert", init=False)
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

class LiteRTInterpreterMetadata(BaseModel):
    """
    Metadata to compile a LiteRT `Interpreter` for inference.

    Members:
        interpreter (tensorflow.lite.Interpreter | ai_edge_litert.interpreter.Interpreter): LiteRT interpreter.
        model_path (str | Path): TFLite model path. The model must exist at this path in the compiler sandbox.
    """
    kind: Literal["meta.inference.tflite"] = Field(default="meta.inference.tflite", init=False)
    interpreter: Annotated[object, BeforeValidator(_validate_litert_interpreter)] = Field(
        description="LiteRT interpreter to apply metadata to.",
        exclude=True
    )
    model_path: str | Path = Field(
        description="TFLite model path. The model must exist at this path in the compiler sandbox.",
        exclude=True
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)