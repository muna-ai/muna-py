# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from inspect import isfunction, signature
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import Annotated, Literal

def _validate_tokenize_function(func):
    if not isfunction(func):
        raise ValueError("`tokenize` must be a plain function.")
    if "." in func.__qualname__:
        raise ValueError("`tokenize` must be a module-level function.")
    sig = signature(func)
    for name, param in sig.parameters.items():
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            raise ValueError(f"`tokenize` parameter '{name}' must be a named parameter (no *args/**kwargs/positional-only).")
    return func

class KVRoutingMetadata(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, frozen=True)
):
    """
    Metadata to compile a tokenization sidecar for KV cache-aware routing.

    The `tokenize` function must map a subset of the predictor's parameters
    to the exact prompt the predictor computes internally: either the prompt
    token IDs, or a processor output (e.g. a `BatchFeature`) whose
    `input_ids` is the prompt. Processor outputs let image prompts route on
    image content. Pass the same function object the predictor calls, so the
    two cannot drift.

    Members:
        tokenize (Callable[..., list[int] | BatchFeature]): Function mapping predictor inputs to prompt token IDs or processor outputs.
    """
    kind: Literal["meta.routing.kv"] = Field("meta.routing.kv", init=False)
    tokenize: Annotated[object, BeforeValidator(_validate_tokenize_function)] = Field(
        description="Function mapping predictor inputs to prompt token IDs, or to processor outputs whose `input_ids` is the prompt.",
        exclude=True
    )