# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ._ort import OnnxRuntimeInferenceSessionMetadataBase
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

class _TensorRTInferenceMetadataBase(BaseModel, **ConfigDict(frozen=True)):
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

class TensorRTInferenceMetadata(_TensorRTInferenceMetadataBase, PyTorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference on Nvidia GPUs with TensorRT.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        optimum_config (optimum.ExporterConfig): Optimum exporter configuration. Required when `exporter` is `optimum`.
        cuda_arch (CudaArchitecture): Target CUDA architecture for the TensorRT engine. Defaults to `sm_80` (Ampere).
        precision (TensorRTPrecision): TensorRT engine inference precision. Defaults to `fp16`.
        hardware_compatibility (TensorRTHardwareCompatibility): TensorRT engine hardware compatibility. Defaults to `none`.
    """
    kind: Literal["meta.inference.tensorrt"] = Field(default="meta.inference.tensorrt", init=False)

class TensorRTInferenceSessionMetadata(_TensorRTInferenceMetadataBase, OnnxRuntimeInferenceSessionMetadataBase):
    """
    Metadata to compile an OnnxRuntime `InferenceSession` for inference on Nvidia GPUs with TensorRT.

    Members:
        session (onnxruntime.InferenceSession): OnnxRuntime inference session to apply metadata to.
        model_path (str | Path): ONNX model path. The file must exist in the compiler sandbox.
        external_data_path (str | Path): ONNX model external data path. This file must exist in the compiler sandbox.
        cuda_arch (CudaArchitecture): Target CUDA architecture for the TensorRT engine. Defaults to `sm_80` (Ampere).
        precision (TensorRTPrecision): TensorRT engine inference precision. Defaults to `fp16`.
        hardware_compatibility (TensorRTHardwareCompatibility): TensorRT engine hardware compatibility. Defaults to `none`.
    """
    kind: Literal["meta.inference.tensorrt_onnx"] = Field(default="meta.inference.tensorrt_onnx", init=False)