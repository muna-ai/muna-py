# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import PyTorchInferenceMetadataBase

QnnInferenceBackend = Literal["cpu", "gpu", "htp"]
QnnInferenceQuantization = Literal["w8a8", "w8a16", "w4a8", "w4a16"]

class QnnInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference on Qualcomm accelerators with QNN SDK.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        model_args (tuple[Tensor,...]): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        output_keys (list): Model output dictionary keys. Use this if the model returns a dictionary.
        exporter (TorchExporter): PyTorch exporter to use.
        backend (QnnInferenceBackend): QNN inference backend. Defaults to `cpu`.
        quantization (QnnInferenceQuantization): QNN model quantization mode. This MUST only be specified when backend is `htp`.
    """
    kind: Literal["meta.inference.qnn"] = Field(default="meta.inference.qnn", init=False)
    backend: QnnInferenceBackend = Field(
        default="cpu",
        description="QNN backend to execute the model.",
        exclude=True
    )
    quantization: QnnInferenceQuantization | None = Field(
        default=None,
        description="QNN model quantization mode. This MUST only be specified when backend is `htp`.",
        exclude=True
    )