# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable
from datetime import timedelta
from functools import wraps
from inspect import isasyncgenfunction, iscoroutinefunction
from math import ceil
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from re import fullmatch, finditer
from types import ModuleType
from typing import Callable, ParamSpec, TypeVar, cast

from .beta import CompileDialect, CompileMetadata, CompileTarget
from .sandbox import Sandbox
from .types import PredictorAccess, PredictorTenancy
from .types.predictor import PredictorDistribution

P = ParamSpec("P")
R = TypeVar("R")

class PredictorSpec(
    BaseModel,
    **ConfigDict(arbitrary_types_allowed=True, extra="allow")
):
    """
    Descriptor of a predictor to be compiled.
    """
    tag: str | None = Field(description="Predictor tag.")
    description: str | None = Field(
        description="Predictor description. MUST be less than 100 characters long.",
        min_length=4,
        max_length=100
    )
    sandbox: Sandbox = Field(description="Sandbox to compile the function.")
    targets: list[CompileTarget] | None = Field(
        default=None,
        description="Targets to compile this predictor for."
    )
    metadata: list[CompileMetadata] | None = Field(
        default=None,
        description="Metadata to use when compiling the function."
    )
    dialects: list[CompileDialect] | None = Field(
        default=None,
        description="Custom dialects to use when compiling the function."
    )
    access: PredictorAccess | None = Field(
        default=None,
        description="Predictor access."
    )
    distribution: PredictorDistribution | None = Field(
        default=None,
        description="Predictor distribution."
    )
    tenancy: PredictorTenancy | None = Field(
        default=None,
        description="Whether the predictor shares GPUs with other predictors, or gets full GPU performance."
    )
    max_queue_time: timedelta | None = Field(
        default=None,
        description="Maximum time a request waits for compute before it is rejected. `None` uses the Muna default.",
        serialization_alias="maxQueueTime"
    )

    @field_validator("max_queue_time", mode="before")
    @classmethod
    def _parse_max_queue_time(cls, value: object) -> timedelta | None:
        if value is None:
            return None
        if isinstance(value, timedelta):
            duration = value
        elif isinstance(value, str):
            duration = _parse_duration(value)
        else:
            # Bare numbers are rejected so that the unit, and opting into batch, is always explicit
            raise ValueError(
                f"`max_queue_time` must be a `timedelta` or a duration string like "
                f"'30s', '10m', '24h', or '1d', not {value!r}"
            )
        if duration <= timedelta(0):
            raise ValueError(f"`max_queue_time` must be positive, not {value!r}")
        return duration

    @field_serializer("max_queue_time")
    def _serialize_max_queue_time(self, value: timedelta | None) -> int | None:
        return ceil(value.total_seconds()) if value is not None else None

def compile(
    *,
    tag: str | None = None,
    description: str | None = None,
    sandbox: Sandbox = None,
    trace_modules: list[ModuleType] = [],
    targets: list[CompileTarget] | None = None,
    metadata: list[CompileMetadata] | None = None,
    dialects: list[CompileDialect] | None = None,
    access: PredictorAccess | None = None,
    distribution: PredictorDistribution | None = None,
    tenancy: PredictorTenancy | None = None,
    max_queue_time: timedelta | str | None = None,
    **kwargs
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Create a predictor by compiling a stateless function.

    Parameters:
        tag (str): Predictor tag.
        description (str): Predictor description. MUST be less than 100 characters long.
        sandbox (Sandbox): Sandbox to compile the function.
        trace_modules (list): Modules to trace and compile.
        targets (list): Targets to compile this predictor for. Pass `None` to compile for our default targets.
        metadata (list): Metadata to use when compiling the function.
        dialects (list): Custom dialects to use when compiling the function.
        access (PredictorAccess): Predictor access.
        distribution (PredictorDistribution): Predictor distribution.
        tenancy (PredictorTenancy): Predictor tenancy.
        max_queue_time (timedelta | str): Maximum time a request waits for compute before it is rejected.
    """
    def decorator(func: Callable):
        # Check type
        if not callable(func):
            raise TypeError("Cannot compile non-function objects")
        if isasyncgenfunction(func) or iscoroutinefunction(func):
            raise TypeError(f"Entrypoint function '{func.__name__}' must be a regular function or generator")            
        # Gather metadata
        spec = PredictorSpec(
            tag=tag,
            description=description,
            sandbox=sandbox if sandbox is not None else Sandbox(),
            targets=targets,
            access=access,
            distribution=distribution,
            tenancy=tenancy,
            max_queue_time=max_queue_time,
            trace_modules=trace_modules,
            metadata=metadata,
            dialects=dialects,
            **kwargs
        )
        # Wrap
        @wraps(func)
        def wrapper (*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__predictor_spec = spec
        return cast(Callable[P, R], wrapper)
    return decorator

def _parse_duration(value: str) -> timedelta:
    """
    Parse a duration string like '30s', '10m', '24h', '1d', or '1h30m'.
    """
    pattern = r"(\d+(?:\.\d+)?)\s*([smhd])"
    text = value.strip().lower()
    if not fullmatch(rf"(?:{pattern}\s*)+", text):
        raise ValueError(
            f"`max_queue_time` must be a duration string like '30s', '10m', '24h', or '1d', not {value!r}"
        )
    seconds = sum(
        float(amount) * _DURATION_UNITS[unit]
        for amount, unit in (m.groups() for m in finditer(pattern, text))
    )
    return timedelta(seconds=seconds)

_DURATION_UNITS = { "s": 1, "m": 60, "h": 3600, "d": 86400 }