# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
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
        model_args (tuple[Tensor,...]): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        output_keys (list): Model output dictionary keys. Use this if the model returns a dictionary.
        exporter (TorchExporter): PyTorch exporter to use.
        precision (TensorRTPrecision): TensorRT engine inference precision. Defaults to `fp16`.
    """
    kind: Literal["meta.inference.tensorrt_rtx"] = Field(default="meta.inference.tensorrt_rtx", init=False)
    precision: TensorRTPrecision = Field(
        default="fp32",
        description="TensorRT engine inference precision.",
        exclude=True
    )