# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

TorchExporter = Literal["none", "dynamo", "torchscript", "optimum"]

def _validate_torch_module(module: "torch.nn.Module") -> "torch.nn.Module": # type: ignore
    try:
        from torch.nn import Module
        if not isinstance(module, Module):
            raise ValueError(f"Expected `torch.nn.Module` model but got `{type(module).__qualname__}`")
        return module
    except ImportError:
        raise ImportError("PyTorch is required to create this metadata but it is not installed.")
    
def _validate_torch_tensor_args(args: list) -> list | None:
    if args is None:
        return args
    try:
        from torch import Tensor
        for idx, arg in enumerate(args):
            if not isinstance(arg, Tensor):
                raise ValueError(f"Expected `torch.Tensor` instance at `model_args[{idx}]` but got `{type(arg).__qualname__}`")
        return args
    except ImportError:
        raise ImportError("PyTorch is required to create this metadata but it is not installed.")

def _validate_optimum_exporter_config(config: object) -> object:
    if config is None:
        return config
    try:
        from optimum.exporters.base import ExporterConfig
        if not isinstance(config, ExporterConfig):
            raise ValueError(f"Expected `optimum.exporters.base.ExporterConfig` instance for `optimum_config` but got `{type(config).__qualname__}`")
        return config
    except ImportError:
        pass

class PyTorchInferenceMetadataBase(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    model: Annotated[object, BeforeValidator(_validate_torch_module)] = Field(
        description="PyTorch module to apply metadata to.",
        exclude=True
    )
    exporter: TorchExporter | None = Field(
        default=None,
        description="PyTorch exporter to use.",
        exclude=True
    )
    model_args: Annotated[list[object] | None, BeforeValidator(_validate_torch_tensor_args)] = Field(
        default=None,
        description="Positional inputs to the model. Required except when `exporter` is `optimum`.",
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
    optimum_config: Annotated[object | None, BeforeValidator(_validate_optimum_exporter_config)] = Field(
        default=None,
        description="Optimum exporter configuration. Required when `exporter` is `optimum`.",
        exclude=True
    )