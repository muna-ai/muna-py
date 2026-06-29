# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ._torch import TorchInferenceMetadataBase

LiteRTInterpreterOptions = Literal[
    # XNNPACK (always on)
    "xnnpack_force_fp16",
    "xnnpack_slow_consistent_arithmetic",
    "xnnpack_subgraph_reshaping",
    # CoreML delegate (iOS / macOS).
    "coreml",
    "coreml_neural_engine_only",
    # GPU delegate (Android).
    "gpu",
    "gpu_min_latency",
    "gpu_sustained_speed",
    "gpu_disallow_precision_loss",
    # NNAPI delegate (Android)
    "nnapi",
    "nnapi_disallow_fp16",
]

class TorchToLiteRTInferenceMetadata(TorchInferenceMetadataBase):
    """
    Metadata to compile a PyTorch model for inference with LiteRT.

    Members:
        model (torch.nn.Module): PyTorch module to apply metadata to.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        targets (list | None): Compile targets where this metadata applies.
    """
    kind: Literal["meta.inference.litert"] = Field("meta.inference.litert", init=False)
    exporter: None = Field(default=None, init=False, exclude=True)
    options: list[LiteRTInterpreterOptions] | None = Field(
        default=None,
        description="TFLite interpreter options.",
        exclude=True
    )