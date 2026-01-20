# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout, redirect_stderr
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import reduce, wraps, WRAPPER_ASSIGNMENTS
from inspect import signature, Parameter, Signature
from io import StringIO
from pydantic import BaseModel, Field
from secrets import choice
from time import perf_counter
from traceback import format_exc
from typing import Callable, ParamSpec, TypeVar

from .remote import _create_remote_value, _parse_remote_value
from .schema import RemoteAcceleration, RemotePrediction, RemoteValue

P = ParamSpec("P")
R = TypeVar("R")

_prediction_request: ContextVar[CreatePredictionInput | None] = ContextVar(
    "prediction_request", 
    default=None
)

def prediction_endpoint() -> Callable[
    [Callable[P, R]],
    Callable[[CreatePredictionInput], RemotePrediction | Iterator[RemotePrediction]]
]:
    """
    Wrap a function to handle serving remote prediction requests.
    """
    def decorator(func: Callable[P, R]) -> Callable[[CreatePredictionInput], RemotePrediction | Iterator[RemotePrediction]]:
        # Get function signature to determine required parameters
        sig = signature(func)
        required_params = {
            name for name, param in sig.parameters.items()
            if param.default is Parameter.empty
            and param.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        }
        # Define wrapper
        @wraps(
            func,
            assigned=(a for a in WRAPPER_ASSIGNMENTS if a != "__annotations__")
        )
        def wrapper(input: CreatePredictionInput) -> RemotePrediction | Iterator[RemotePrediction]:
            if input.stream:
                return _invoke_streaming(
                    func=func,
                    input=input,
                    required_params=required_params
                )
            else:
                return _invoke_eager(
                    func=func,
                    input=input,
                    required_params=required_params
                )
        # Explicitly set signature with concrete types (vs. forward references)
        # This is done for compatibility with FastAPI.
        wrapper.__signature__ = Signature(
            parameters=[
                Parameter(
                    name="input",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=CreatePredictionInput
                )
            ],
            return_annotation=RemotePrediction
        )
        # Return
        return wrapper
    return decorator

def get_prediction_request() -> CreatePredictionInput | None:
    """
    Get the current prediction request, or `None` if not in scope.
    """
    return _prediction_request.get()

def _invoke_eager(
    *,
    func: Callable,
    input: CreatePredictionInput,
    required_params: set[str]
) -> RemotePrediction:
    prediction_id = _create_prediction_id()
    stdout_buffer = StringIO()
    token = _prediction_request.set(input)
    start_time = perf_counter()
    try:
        kwargs = _parse_inputs(input.inputs, required_params)
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            result = func(**kwargs)
        if isinstance(result, Iterator):
            result = reduce(lambda _, x: x, result)
        return _create_prediction(
            id=prediction_id,
            tag=input.tag,
            results=result,
            start_time=start_time,
            logs=stdout_buffer,
            created=datetime.now(timezone.utc).isoformat()
        )
    except Exception:
        latency = (perf_counter() - start_time) * 1000
        return RemotePrediction(
            id=prediction_id,
            tag=input.tag,
            latency=latency,
            logs=stdout_buffer.getvalue(),
            error=format_exc(),
            created=datetime.now(timezone.utc).isoformat()
        )
    finally:
        _prediction_request.reset(token)

def _invoke_streaming(
    *,
    func: Callable,
    input: CreatePredictionInput,
    required_params: set[str]
) -> Iterator[RemotePrediction]:
    prediction_id = _create_prediction_id()
    stdout_buffer = StringIO()
    token = _prediction_request.set(input)
    start_time = perf_counter()
    try:
        kwargs = _parse_inputs(input.inputs, required_params)
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            result = func(**kwargs)
        created = datetime.now(timezone.utc).isoformat()
        def stream_with_context():
            try:
                stream = result if isinstance(result, Iterator) else iter([result])
                for r in stream:
                    yield _create_prediction(
                        id=prediction_id,
                        tag=input.tag,
                        results=r,
                        start_time=start_time,
                        logs=stdout_buffer,
                        created=created
                    )
            except Exception:
                latency = (perf_counter() - start_time) * 1000
                yield RemotePrediction(
                    id=prediction_id,
                    tag=input.tag,
                    latency=latency,
                    logs=stdout_buffer.getvalue(),
                    error=format_exc(),
                    created=datetime.now(timezone.utc).isoformat()
                )
            finally:
                _prediction_request.reset(token)
        return stream_with_context()
    except Exception:
        _prediction_request.reset(token)
        latency = (perf_counter() - start_time) * 1000
        prediction = RemotePrediction(
            id=prediction_id,
            tag=input.tag,
            latency=latency,
            logs=stdout_buffer.getvalue(),
            error=format_exc(),
            created=datetime.now(timezone.utc).isoformat()
        )
        return iter([prediction])

def _parse_inputs(
    inputs: dict[str, RemoteValue],
    required_params: set[str],
) -> dict[str, object]:
    missing_args = required_params - set(inputs.keys())
    if missing_args:
        arg_name = next(iter(missing_args))
        raise ValueError(
            f"Failed to create prediction because required "
            f"input argument `{arg_name}` was not provided."
        )
    return {
        name: _parse_remote_value(value)
        for name, value in inputs.items()
    }

def _create_prediction(
    *,
    id: str,
    tag: str,
    results: object,
    start_time: float,
    logs: StringIO,
    created: str
) -> RemotePrediction:
    latency = (perf_counter() - start_time) * 1000 # millis
    results = list(results) if isinstance(results, tuple) else [results]
    result_values = [_create_remote_value(value) for value in results]
    return RemotePrediction(
        id=id,
        tag=tag,
        results=result_values,
        latency=latency,
        logs=logs.getvalue(),
        created=created
    )

def _create_prediction_id() -> str:
    ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    random = "".join(choice(ALPHABET) for _ in range(21))
    return f"pred_{random}"

class CreatePredictionInput(BaseModel):
    tag: str = Field(description="Predictor tag.")
    inputs: dict[str, RemoteValue] = Field(description="Prediction inputs.")
    api_url: str | None = Field(default=None, description="Muna API URL.")
    access_key: str | None = Field(default=None, description="Muna access key.")
    acceleration: RemoteAcceleration | str | None = Field(default=None, description="Prediction acceleration.")
    stream: bool = Field(default=False, description="Whether to stream predictions.")