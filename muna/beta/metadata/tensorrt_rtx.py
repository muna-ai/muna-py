# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import TorchInferenceMetadataBase

class TorchToTensorRTRTXInferenceMetadata(TorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference on Nvidia RTX GPUs with TensorRT-RTX.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        targets (list | None): Compile targets where this metadata applies.
    """
    kind: Literal["meta.inference.tensorrt_rtx"] = Field("meta.inference.tensorrt_rtx", init=False)