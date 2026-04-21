# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

from ._torch import _validate_torch_module, _validate_torch_tensor_args

CudaArchitecture = Literal[
    "sm_80",    # Ampere
    "sm_80+",   # Ampere or newer
    "sm_86",    # Ampere
    "sm_89",    # Ada Lovelace
    "sm_90",    # Hopper
    "sm_100",   # Blackwell
]

class GeneratedCudaKernelInferenceMetadata(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    """
    Metadata to compile a PyTorch model using an AI-generated CUDA kernel.

    Members:
        model (torch.nn.Module): PyTorch model to compile.
        model_args (tuple): Positional inputs to the model.
        input_shapes (list): Model input tensor shapes. Use this to specify dynamic axes.
        cuda_arch (CudaArchitecture): Target CUDA architecture for the generated kernel. Defaults to `sm_80+` (Ampere or newer).
        prompt (str): Custom instructions to steer kernel generation.
    """
    kind: Literal["meta.inference.generated.cuda"] = Field(default="meta.inference.generated.cuda", init=False)
    model: Annotated[object, BeforeValidator(_validate_torch_module)] = Field(
        description="PyTorch model to compile.",
        exclude=True
    )
    model_args: Annotated[list[object] | None, BeforeValidator(_validate_torch_tensor_args)] = Field(
        default=None,
        description="Positional inputs to the model. Used for testing generated kernel correctness.",
        exclude=True
    )
    input_shapes: list[tuple] | None = Field(
        default=None,
        description="Model input tensor shapes. Use this to specify dynamic axes.",
        exclude=True
    )
    cuda_arch: CudaArchitecture = Field(
        default="sm_80+",
        description="Target CUDA architecture for the generated kernel.",
        exclude=True
    )
    prompt: str | None = Field(
        default=None,
        description="""
        Custom instructions to steer kernel generation. Use this to express stylistic 
        or compositional preferences (e.g. preferred libraries, fusion choices).
        """,
        exclude=True
    )