# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Annotated

from ._ort import OnnxRuntimeExecutionProvider, OnnxRuntimeOptimizationLevel
from ._speculative import SpeculativeDecodingConfig
from ._torch import TorchExporter
from .coreml import (
    CoreMLComputeUnit, OnnxRuntimeToCoreMLInferenceMetadata,
    TorchToCoreMLInferenceMetadata
)
from .executorch import ExecuTorchInferenceBackend, ExecuTorchInferenceMetadata
from .litert import LiteRTInterpreterOptions, TorchToLiteRTInferenceMetadata
from .llama import LlamaCppBackend, LlamaCppInferenceMetadata
from .mlx import OnnxRuntimeToMLXInferenceMetadata, TorchToMLXInferenceMetadata
from .onnxruntime import OnnxRuntimeInferenceSessionMetadata, TorchToOnnxRuntimeInferenceMetadata
from .openvino import TorchToOpenVINOInferenceMetadata
from .qnn import (
    QnnInferenceBackend, QnnInferenceQuantization,
    TorchToQnnInferenceMetadata
)
from .routing import KVRoutingMetadata
from .sglang import (
    DiffusersToSGLangInferenceMetadata, SGLangComputeArchitecture,
    SGLangDisaggregationConfig, TorchToSGLangInferenceMetadata
)
from .tensorrt import (
    CudaArchitecture, TorchToTensorRTInferenceMetadata,
    OnnxRuntimeToTensorRTInferenceMetadata
)
from .tensorrt_rtx import TorchToTensorRTRTXInferenceMetadata
from .tflite import TFLiteInterpreterMetadata

CompileMetadata = Annotated[
    # PyTorch
    ExecuTorchInferenceMetadata             |
    TorchToCoreMLInferenceMetadata          |
    TorchToLiteRTInferenceMetadata          |
    TorchToMLXInferenceMetadata             |
    TorchToOnnxRuntimeInferenceMetadata     |
    TorchToOpenVINOInferenceMetadata        |
    TorchToQnnInferenceMetadata             |
    TorchToTensorRTInferenceMetadata        |
    TorchToTensorRTRTXInferenceMetadata     |
    # Transformers
    TorchToSGLangInferenceMetadata          |
    # Diffusers
    DiffusersToSGLangInferenceMetadata      |
    # ONNX
    OnnxRuntimeInferenceSessionMetadata     |
    OnnxRuntimeToCoreMLInferenceMetadata    |
    OnnxRuntimeToMLXInferenceMetadata       |
    OnnxRuntimeToTensorRTInferenceMetadata  |
    # Routing
    KVRoutingMetadata                       |
    # Misc
    LlamaCppInferenceMetadata               |
    TFLiteInterpreterMetadata,
    Field(discriminator="kind")
]

# Deprecated aliases
CoreMLInferenceMetadata = TorchToCoreMLInferenceMetadata
LiteRTInferenceMetadata = TorchToLiteRTInferenceMetadata
MLXInferenceMetadata = TorchToMLXInferenceMetadata
MLXInferenceSessionMetadata = OnnxRuntimeToMLXInferenceMetadata
OnnxRuntimeInferenceMetadata = TorchToOnnxRuntimeInferenceMetadata
OpenVINOInferenceMetadata = TorchToOpenVINOInferenceMetadata
QnnInferenceMetadata = TorchToQnnInferenceMetadata
SGLangInferenceMetadata = TorchToSGLangInferenceMetadata
TensorRTRTXInferenceMetadata = TorchToTensorRTRTXInferenceMetadata
TensorRTInferenceMetadata = TorchToTensorRTInferenceMetadata
TensorRTInferenceSessionMetadata = OnnxRuntimeToTensorRTInferenceMetadata