# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import model_validator, BaseModel, ConfigDict, Field
from typing import Literal

from ._ort import OnnxRuntimeInferenceSessionMetadataBase
from ._torch import TorchInferenceMetadataBase

CoreMLComputeUnit = Literal["cpu", "gpu", "ane"]
CoreMLComputePrecision = Literal["fp16", "fp32"]

class _CoreMLInferenceMetadataBase(
    BaseModel,
    **ConfigDict(frozen=True, arbitrary_types_allowed=True)
):
    compute_units: list[CoreMLComputeUnit] | None = Field(
        default=None,
        description="CoreML compute units.",
        exclude=True,
    )
    compute_precision: CoreMLComputePrecision | None = Field(
        default=None,
        description="CoreML compute precision.",
        exclude=True
    )
    ios_deployment_target: int | None = Field(
        default=None,
        description="Minimum iOS deployment target.",
        exclude=True,
        ge=13,
        le=26
    )
    macos_deployment_target: int | None = Field(
        default=None,
        description="Minimum macOS deployment target.",
        exclude=True,
        ge=11,
        le=26
    )

    @model_validator(mode="after")
    def _validate_deployment_target(self):
        if self.ios_deployment_target is not None and self.macos_deployment_target is not None:
            raise ValueError("Specify either `ios_deployment_target` or `macos_deployment_target`, not both.")
        return self

class TorchToCoreMLInferenceMetadata(
    _CoreMLInferenceMetadataBase,
    TorchInferenceMetadataBase
):
    """
    Metadata to compile a PyTorch model for inference with CoreML.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        exporter (TorchExporter): PyTorch exporter to use.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list | None): Model input tensor shapes. Use this to specify dynamic axes.
        targets (list | None): Compile targets where this metadata applies.
        compute_units (list | None): CoreML compute units.
        compute_precision (CoreMLComputePreision): CoreML compute precision.
        ios_deployment_target (int | None): Minimum iOS deployment target.
        macos_deployment_target (int | None): Minimum macOS deployment target.
    """
    kind: Literal["meta.inference.coreml"] = Field("meta.inference.coreml", init=False)

class OnnxRuntimeToCoreMLInferenceMetadata(
    _CoreMLInferenceMetadataBase,
    OnnxRuntimeInferenceSessionMetadataBase
):
    """
    Metadata to compile an ONNXRuntime `InferenceSession` for inference with CoreML.

    Members:
        session (onnxruntime.InferenceSession): OnnxRuntime inference session to apply metadata to.
        model_path (str | Path): ONNX model path. The file must exist in the compiler sandbox.
        external_data_path (str | Path): ONNX model external data path. This file must exist in the compiler sandbox.
        targets (list | None): Compile targets where this metadata applies.
        compute_units (list): CoreML compute units.
        compute_precision (CoreMLComputePreision): CoreML compute precision.
        ios_deployment_target (int | None): Minimum iOS deployment target.
        macos_deployment_target (int | None): Minimum macOS deployment target.
    """
    kind: Literal["meta.inference.onnx_coreml"] = Field("meta.inference.onnx_coreml", init=False)