# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ._ort import OnnxRuntimeInferenceSessionMetadataBase
from ._torch import TorchInferenceMetadataBase

CudaArchitecture = Literal[
    "sm_80",    # Ampere
    "sm_80+",   # Ampere or newer
    "sm_86",    # Ampere
    "sm_89",    # Ada Lovelace
    "sm_90",    # Hopper
    "sm_100",   # Blackwell
]

class _TensorRTInferenceMetadataBase(BaseModel, **ConfigDict(frozen=True)):
    cuda_arch: CudaArchitecture = Field(
        default="sm_80+",
        description="Target CUDA architecture for the TensorRT engine.",
        exclude=True
    )

class TorchToTensorRTInferenceMetadata(
    _TensorRTInferenceMetadataBase,
    TorchInferenceMetadataBase
):
    """
    Metadata to compile a PyTorch model for inference on Nvidia GPUs with TensorRT.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        targets (list | None): Compile targets where this metadata applies.
        cuda_arch (CudaArchitecture): Target CUDA architecture for the TensorRT engine. Defaults to `sm_80+` (Ampere or newer).
        hardware_compatibility (TensorRTHardwareCompatibility): TensorRT engine hardware compatibility. Defaults to `forward_compatibility`.
    """
    kind: Literal["meta.inference.tensorrt"] = Field(default="meta.inference.tensorrt", init=False)

class OnnxRuntimeToTensorRTInferenceMetadata(
    _TensorRTInferenceMetadataBase,
    OnnxRuntimeInferenceSessionMetadataBase
):
    """
    Metadata to compile an ONNXRuntime `InferenceSession` for inference on Nvidia GPUs with TensorRT.

    Members:
        session (onnxruntime.InferenceSession): ONNXRuntime inference session to apply metadata to.
        model_path (str | Path): ONNX model path. The file must exist in the compiler sandbox.
        external_data_path (str | Path): ONNX model external data path. This file must exist in the compiler sandbox.
        targets (list | None): Compile targets where this metadata applies.
        cuda_arch (CudaArchitecture): Target CUDA architecture for the TensorRT engine. Defaults to `sm_80+` (Ampere or newer).
        hardware_compatibility (TensorRTHardwareCompatibility): TensorRT engine hardware compatibility. Defaults to `forward_compatibility`.
    """
    kind: Literal["meta.inference.tensorrt_onnx"] = Field(default="meta.inference.tensorrt_onnx", init=False)