# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import PyTorchInferenceMetadataBase

class LiteRTInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference with LiteRT.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
    """
    kind: Literal["meta.inference.litert"] = Field(default="meta.inference.litert", init=False)
    exporter: None = Field(default=None, init=False, exclude=True)