# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

from ._torch import _validate_torch_module
from .tensorrt import CudaArchitecture

TensorRTLLMQuantization = Literal[
    "fp16",
    "bf16",
    "fp8",
    "int8_sq",
    "int4_awq",
    "int4_gptq",
    "nvfp4",
]

TensorRTLLMKVCacheQuantization = Literal["fp8", "int8", "nvfp4"]

class TensorRTLLMInferenceMetadata(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    """
    Metadata to compile a large language model for multi-GPU inference with TensorRT-LLM.

    Members:
        model (torch.nn.Module): Large language model to compile.
        quantization (TensorRTLLMQuantization): Weight/activation quantization format.
        quantization_samples (list[str]): Text samples for quantization calibration.
        kv_cache (TensorRTLLMKVCacheQuantization): KV cache quantization format.
        tensor_parallel (int): Number of GPUs for tensor parallelism.
        max_batch_size (int): Maximum concurrent batch size.
        max_input_len (int): Maximum input sequence length.
        max_output_len (int): Maximum generation length.
        max_num_tokens (int): Maximum total tokens for inflight batching.
        cuda_arch (CudaArchitecture): Target CUDA architecture.
    """
    kind: Literal["meta.inference.tensorrt_llm"] = Field(default="meta.inference.tensorrt_llm", init=False)
    model: Annotated[object, BeforeValidator(_validate_torch_module)] = Field(
        description="Large language model to compile.",
        exclude=True
    )
    quantization: TensorRTLLMQuantization = Field(
        default="fp16",
        description="Weight and activation quantization format for the TensorRT-LLM engine.",
        exclude=True
    )
    quantization_samples: list[str] | None = Field(
        default=None,
        description="Text samples for quantization calibration. Required when quantization is not fp16 or bf16.",
        exclude=True
    )
    kv_cache: TensorRTLLMKVCacheQuantization | None = Field(
        default=None,
        description="KV cache quantization format. When `None`, inherits from model quantization.",
        exclude=True
    )
    tensor_parallel: int = Field(
        default=1,
        description="Number of GPUs for tensor parallelism. Baked into the engine at build time.",
        exclude=True,
        ge=1
    )
    max_batch_size: int = Field(
        default=8,
        description="Maximum concurrent batch size for the engine.",
        exclude=True,
        ge=1
    )
    max_input_len: int = Field(
        default=4096,
        description="Maximum input sequence length.",
        exclude=True,
        ge=1
    )
    max_output_len: int = Field(
        default=2048,
        description="Maximum output sequence length for generation.",
        exclude=True,
        ge=1
    )
    max_num_tokens: int | None = Field(
        default=None,
        description="Maximum total tokens for inflight batching. When `None`, defaults to `max_batch_size * max_input_len`.",
        exclude=True
    )
    cuda_arch: CudaArchitecture = Field(
        default="sm_90",
        description="Target CUDA architecture for the TensorRT-LLM engine.",
        exclude=True
    )