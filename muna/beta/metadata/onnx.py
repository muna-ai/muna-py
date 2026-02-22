# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import PyTorchInferenceMetadataBase

OnnxRuntimeExecutionProvider = Literal["cpu", "coreml", "cuda", "openvino", "xnnpack"]
OnnxRuntimeOptimizationLevel = Literal["none", "basic", "extended"]

class OnnxRuntimeInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference with OnnxRuntime.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple[Tensor,...]): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        output_keys (list): Model output dictionary keys. Use this if the model returns a dictionary.
        optimum_config (optimum.ExporterConfig): Optimum exporter configuration. Required when `exporter` is `optimum`.
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