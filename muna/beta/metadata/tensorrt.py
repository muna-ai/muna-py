# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import PyTorchInferenceMetadataBase

CudaArchitecture = Literal[
    "sm_80",    # Ampere (A100)
    "sm_86",    # Ampere (A10)
    "sm_89",    # Ada Lovelace (L40)
    "sm_90",    # Hopper (H100)
    "sm_100",   # Blackwell (B200)
]

TensorRTPrecision = Literal["bf16", "fp4", "fp8", "fp16", "fp32", "int8"]
TensorRTHardwareCompatibility = Literal["none", "same_compatibility", "forward_compatibility"]

class TensorRTInferenceMetadata(PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference on Nvidia GPUs with TensorRT.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        model_args (tuple[Tensor,...]): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        output_keys (list): Model output dictionary keys. Use this if the model returns a dictionary.
        exporter (TorchExporter): PyTorch exporter to use.
        cuda_arch (CudaArchitecture): Target CUDA architecture for the TensorRT engine. Defaults to `sm_80` (Ampere).
        precision (TensorRTPrecision): TensorRT engine inference precision. Defaults to `fp16`.
        hardware_compatibility (TensorRTHardwareCompatibility): TensorRT engine hardware compatibility. Defaults to `none`.
    """
    kind: Literal["meta.inference.tensorrt"] = Field(default="meta.inference.tensorrt", init=False)
    cuda_arch: CudaArchitecture = Field(
        default="sm_80",
        description="Target CUDA architecture for the TensorRT engine.",
        exclude=True
    )
    precision: TensorRTPrecision = Field(
        default="fp32",
        description="TensorRT engine inference precision.",
        exclude=True
    )
    hardware_compatibility: TensorRTHardwareCompatibility = Field(
        default="forward_compatibility",
        description="TensorRT hardware compatibility mode.",
        exclude=True
    )