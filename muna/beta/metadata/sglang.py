# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

from ._diffusers import validate_diffusers_pipeline
from ._speculative import SpeculativeDecodingConfig
from ._torch import validate_torch_module
from .tensorrt import CudaArchitecture

SGLangComputeArchitecture = CudaArchitecture

class TorchToSGLangInferenceMetadata(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Metadata to compile a large language model for multi-GPU inference with SGLang.

    Members:
        model (torch.nn.Module): Large language model to compile.
        compute_architecture (SGLangComputeArchitecture): Compute architecture which the SGLang engine targets.
        speculative_decoding (SpeculativeDecodingConfig): Speculative decoding configuration.
        max_running_requests (int): Maximum concurrent in-flight requests at the scheduler.
        max_total_tokens (int): Total KV cache capacity.
        tensor_parallelism (int): Tensor parallelism size.
    """
    kind: Literal["meta.inference.sglang"] = Field(default="meta.inference.sglang", init=False)
    model: Annotated[object, BeforeValidator(validate_torch_module)] = Field(
        description="Large language model to compile.",
        exclude=True
    )
    compute_architecture: SGLangComputeArchitecture = Field(
        description="Compute architecture which the SGLang engine targets.",
        exclude=True
    )
    speculative_decoding: SpeculativeDecodingConfig | None = Field(
        default=None,
        description="Speculative decoding configuration.",
        exclude=True
    )
    max_running_requests: int | None = Field(
        default=None,
        description="Maximum concurrent in-flight requests at the scheduler.",
        ge=1,
        exclude=True,
    )
    max_total_tokens: int | None = Field(
        default=None,
        description="Total KV cache capacity.",
        ge=1,
        exclude=True,
    )
    tensor_parallelism: int | None = Field(
        default=None,
        description="Tensor parallelism size.",
        ge=1,
        exclude=True
    )

class DiffusersToSGLangInferenceMetadata(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Metadata to compile a diffusion pipeline for inference with SGLang.

    Members:
        pipeline (diffusers.DiffusionPipeline): Diffusion pipeline.
    """
    kind: Literal["meta.inference.sglang_diffusion"] = Field(default="meta.inference.sglang_diffusion", init=False)
    pipeline: Annotated[object, BeforeValidator(validate_diffusers_pipeline)] = Field(
        description="Diffusers pipeline to compile.",
        exclude=True
    )
    compute_architecture: SGLangComputeArchitecture = Field(
        description="Compute architecture which the SGLang engine targets.",
        exclude=True
    )
    max_batch_size: int | None = Field(
        default=None,
        description="Maximum batch size.",
        ge=1,
        exclude=True
    )