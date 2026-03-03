# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import PyTorchInferenceMetadataBase

ExecuTorchInferenceBackend = Literal["xnnpack", "vulkan"]

class ExecuTorchInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference with ExecuTorch.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        backend (ExecuTorchInferenceBackend): ExecuTorch backend to execute the model.
    """
    kind: Literal["meta.inference.executorch"] = Field(default="meta.inference.executorch", init=False)
    exporter: None = Field(default=None, init=False, exclude=True)
    backend: ExecuTorchInferenceBackend = Field(
        default="xnnpack",
        description="ExecuTorch backend to execute the model.",
        exclude=True
    )