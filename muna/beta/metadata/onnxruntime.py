# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._ort import (
    OnnxRuntimeExecutionProvider, OnnxRuntimeInferenceSessionMetadataBase,
    OnnxRuntimeOptimizationLevel
)
from ._torch import TorchInferenceMetadataBase

class TorchToOnnxRuntimeInferenceMetadata(TorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference with ONNXRuntime.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        targets (list | None): Compile targets where this metadata applies.
        optimization (OnnxRuntimeOptimizationLevel): ONNX model optimization level.
        providers (list): Execution providers that can be used to accelerate inference for this model.
    """
    kind: Literal["meta.inference.onnx"] = Field(default="meta.inference.onnx", init=False)
    optimization: OnnxRuntimeOptimizationLevel = Field(
        default="none",
        description="ONNX model optimization level. Defaults to `none`.",
        exclude=True
    )
    providers: list[OnnxRuntimeExecutionProvider] | None = Field(
        default=None,
        description="ONNXRuntime execution providers to build with.",
        exclude=True
    )

class OnnxRuntimeInferenceSessionMetadata(OnnxRuntimeInferenceSessionMetadataBase):
    """
    Metadata to compile an ONNXRuntime `InferenceSession` for inference.

    Members:
        session (onnxruntime.InferenceSession): ONNXRuntime inference session to apply metadata to.
        model_path (str | Path): ONNX model path. The file must exist in the compiler sandbox.
        external_data_path (str | Path): ONNX model external data path. This file must exist in the compiler sandbox.
        targets (list | None): Compile targets where this metadata applies.
        providers (list): Execution providers that can be used to accelerate inference for this model.
    """
    kind: Literal["meta.inference.onnxruntime"] = Field("meta.inference.onnxruntime", init=False)
    providers: list[OnnxRuntimeExecutionProvider] | None = Field(
        default=None,
        description="ONNXRuntime execution providers to build with.",
        exclude=True
    )