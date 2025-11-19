# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

TorchExporter = Literal["none", "dynamo", "torchscript"]

def _validate_torch_module(module: "torch.nn.Module") -> "torch.nn.Module": # type: ignore
    try:
        from torch.nn import Module
        if not isinstance(module, Module):
            raise ValueError(f"Expected `torch.nn.Module` model but got `{type(module).__qualname__}`")
        return module
    except ImportError:
        raise ImportError("PyTorch is required to create this metadata but it is not installed.")
    
def _validate_torch_tensor_args(args: list) -> list:
    try:
        from torch import Tensor
        for idx, arg in enumerate(args):
            if not isinstance(arg, Tensor):
                raise ValueError(f"Expected `torch.Tensor` instance at `model_args[{idx}]` but got `{type(arg).__qualname__}`")
        return args
    except ImportError:
        raise ImportError("PyTorch is required to create this metadata but it is not installed.")

class PyTorchInferenceMetadataBase(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    model: Annotated[object, BeforeValidator(_validate_torch_module)] = Field(
        description="PyTorch module to apply metadata to.",
        exclude=True
    )
    model_args: Annotated[list[object], BeforeValidator(_validate_torch_tensor_args)] = Field(
        description="Positional inputs to the model.",
        exclude=True
    )
    input_shapes: list[tuple] | None = Field(
        default=None,
        description="Model input tensor shapes. Use this to specify dynamic axes.",
        exclude=True
    )
    output_keys: list[str] | None = Field(
        default=None,
        description="Model output dictionary keys. Use this if the model returns a dictionary.",
        exclude=True
    )
    exporter: TorchExporter | None = Field(
        default=None,
        description="PyTorch exporter to use.",
        exclude=True
    )