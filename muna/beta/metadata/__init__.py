# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ._ort import OnnxRuntimeExecutionProvider, OnnxRuntimeOptimizationLevel
from ._torch import KVCacheConfig, TorchExporter
from .coreml import CoreMLInferenceMetadata
from .cuda import CudaArchitecture, GeneratedCudaKernelInferenceMetadata
from .executorch import ExecuTorchInferenceBackend, ExecuTorchInferenceMetadata
from .iree import IREEInferenceBackend, IREEInferenceMetadata
from .litert import LiteRTInferenceMetadata
from .llama import LlamaCppBackend, LlamaCppInferenceMetadata
from .mlx import MLXInferenceMetadata, MLXInferenceSessionMetadata
from .onnxruntime import OnnxRuntimeInferenceMetadata, OnnxRuntimeInferenceSessionMetadata
from .openvino import OpenVINOInferenceMetadata
from .qnn import QnnInferenceBackend, QnnInferenceMetadata, QnnInferenceQuantization
from .sglang import SGLangInferenceMetadata
from .tensorrt import TensorRTInferenceMetadata, TensorRTInferenceSessionMetadata
from .tensorrt_llm import (
    TensorRTLLMInferenceMetadata, TensorRTLLMQuantization,
    TensorRTLLMKVCacheQuantization
)
from .tensorrt_rtx import TensorRTRTXInferenceMetadata
from .tflite import TFLiteInterpreterMetadata
from .vllm import vLLMInferenceMetadata

InferenceMetadata = (
    CoreMLInferenceMetadata                 |
    ExecuTorchInferenceMetadata             |
    GeneratedCudaKernelInferenceMetadata    |
    IREEInferenceMetadata                   |
    LiteRTInferenceMetadata                 |
    LlamaCppInferenceMetadata               |
    MLXInferenceMetadata                    |
    MLXInferenceSessionMetadata             |
    OnnxRuntimeInferenceMetadata            |
    OnnxRuntimeInferenceSessionMetadata     |
    OpenVINOInferenceMetadata               |
    QnnInferenceMetadata                    |
    SGLangInferenceMetadata                 |
    TensorRTInferenceMetadata               |
    TensorRTInferenceSessionMetadata        |
    TensorRTLLMInferenceMetadata            |
    TensorRTRTXInferenceMetadata            |
    TFLiteInterpreterMetadata               |
    vLLMInferenceMetadata
)