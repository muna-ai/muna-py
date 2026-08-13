# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from inspect import isfunction, signature
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing import get_type_hints, Annotated, Literal

def _validate_tokenize_function(func):
    if not isfunction(func):
        raise ValueError("`tokenize` must be a plain function.")
    if "." in func.__qualname__:
        raise ValueError("`tokenize` must be a module-level function.")
    hints = get_type_hints(func)
    if hints.get("return") != list[int]:
        raise ValueError("`tokenize` must be annotated to return `list[int]` (prompt token IDs).")
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
    to the exact prompt token IDs the predictor computes internally. Pass the
    same function object the predictor calls, so the two cannot drift.

    Members:
        tokenize (Callable[..., list[int]]): Tokenization function mapping predictor inputs to prompt token IDs.
    """
    kind: Literal["meta.routing.kv"] = Field("meta.routing.kv", init=False)
    tokenize: Annotated[object, BeforeValidator(_validate_tokenize_function)] = Field(
        description="Tokenization function mapping predictor inputs to prompt token IDs.",
        exclude=True
    )