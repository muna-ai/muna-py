# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import PyTorchInferenceMetadataBase
from .tensorrt import TensorRTPrecision

class TensorRTRTXInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference on Nvidia RTX GPUs with TensorRT-RTX.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        optimum_config (optimum.ExporterConfig): Optimum exporter configuration. Required when `exporter` is `optimum`.
        precision (TensorRTPrecision): TensorRT engine inference precision. Defaults to `fp16`.
    """
    kind: Literal["meta.inference.tensorrt_rtx"] = Field(default="meta.inference.tensorrt_rtx", init=False)
    precision: TensorRTPrecision = Field(
        default="fp32",
        description="TensorRT engine inference precision.",
        exclude=True
    )