# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ._ort import OnnxRuntimeExecutionProvider, OnnxRuntimeOptimizationLevel
from ._speculative import (
    DFlashSpeculativeDecoding, Eagle3SpeculativeDecoding,
    SpeculativeDecodingConfig
)
from ._torch import TorchExporter
from .coreml import (
    CoreMLComputeUnit, OnnxRuntimeToCoreMLInferenceMetadata,
    TorchToCoreMLInferenceMetadata
)
from .executorch import ExecuTorchInferenceBackend, ExecuTorchInferenceMetadata
from .litert import TorchToLiteRTInferenceMetadata
from .llama import LlamaCppBackend, LlamaCppInferenceMetadata
from .mlx import OnnxRuntimeToMLXInferenceMetadata, TorchToMLXInferenceMetadata
from .onnxruntime import OnnxRuntimeInferenceSessionMetadata, TorchToOnnxRuntimeInferenceMetadata
from .openvino import TorchToOpenVINOInferenceMetadata
from .qnn import QnnInferenceBackend, QnnInferenceQuantization, TorchToQnnInferenceMetadata
from .sglang import TorchToSGLangInferenceMetadata
from .tensorrt import (
    CudaArchitecture, TorchToTensorRTInferenceMetadata,
    OnnxRuntimeToTensorRTInferenceMetadata
)
from .tensorrt_rtx import TorchToTensorRTRTXInferenceMetadata
from .tflite import TFLiteInterpreterMetadata
from .vllm import TorchTovLLMInferenceMetadata

CompileMetadata = (
    # PyTorch
    ExecuTorchInferenceMetadata             |
    TorchToCoreMLInferenceMetadata          |
    TorchToLiteRTInferenceMetadata          |
    TorchToMLXInferenceMetadata             |
    TorchToOnnxRuntimeInferenceMetadata     |
    TorchToOpenVINOInferenceMetadata        |
    TorchToQnnInferenceMetadata             |
    TorchToSGLangInferenceMetadata          |
    TorchToTensorRTInferenceMetadata        |
    TorchToTensorRTRTXInferenceMetadata     |
    TorchTovLLMInferenceMetadata            |
    # ONNX
    OnnxRuntimeInferenceSessionMetadata     |
    OnnxRuntimeToCoreMLInferenceMetadata    |
    OnnxRuntimeToMLXInferenceMetadata       |
    OnnxRuntimeToTensorRTInferenceMetadata  |
    # Misc
    LlamaCppInferenceMetadata               |
    TFLiteInterpreterMetadata
)

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