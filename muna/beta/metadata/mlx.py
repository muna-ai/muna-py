# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._ort import OnnxRuntimeInferenceSessionMetadataBase
from ._torch import PyTorchInferenceMetadataBase

class MLXInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference on Apple Silicon with MLX.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        optimum_config (optimum.ExporterConfig): Optimum exporter configuration. Required when `exporter` is `optimum`.
    """
    kind: Literal["meta.inference.mlx"] = Field(default="meta.inference.mlx", init=False)

class MLXInferenceSessionMetadata(OnnxRuntimeInferenceSessionMetadataBase):
    """
    Metadata to compile an OnnxRuntime `InferenceSession` for inference on Apple Silicon with MLX.

    Members:
        session (onnxruntime.InferenceSession): OnnxRuntime inference session to apply metadata to.
        model_path (str | Path): ONNX model path. The file must exist in the compiler sandbox.
        external_data_path (str | Path): ONNX model external data path. This file must exist in the compiler sandbox.
    """
    kind: Literal["meta.inference.mlx_onnx"] = Field(default="meta.inference.mlx_onnx", init=False)