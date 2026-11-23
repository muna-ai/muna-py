# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ..compile import CompileTarget

TorchExporter = Literal["none", "dynamo", "torchscript"]

class TorchInferenceMetadataBase(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    model: object = Field(
        description="PyTorch module to apply metadata to.",
        exclude=True
    )
    exporter: TorchExporter | None = Field(
        default=None,
        description="PyTorch exporter to use.",
        exclude=True
    )
    model_args: list[object] | None = Field(
        default=None,
        description="Positional inputs to the model.",
        exclude=True
    )
    input_shapes: list[tuple] | None = Field(
        default=None,
        description="Model input tensor shapes. Use this to specify dynamic axes.",
        exclude=True
    )
    targets: list[CompileTarget] | None = Field(
        default=None,
        description="Compile targets where this metadata should apply.",
        exclude=True
    )