# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import PyTorchInferenceMetadataBase

IREEInferenceBackend = Literal["vulkan"]

class IREEInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference with IREE.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        model_args (tuple[Tensor,...]): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        output_keys (list): Model output dictionary keys. Use this if the model returns a dictionary.
        exporter (TorchExporter): PyTorch exporter to use.
    """
    kind: Literal["meta.inference.iree"] = Field(default="meta.inference.iree", init=False)
    backend: IREEInferenceBackend = Field(
        default="vulkan",
        description="IREE HAL target backend to execute the model.",
        exclude=True
    )