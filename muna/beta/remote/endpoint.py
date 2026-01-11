# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout, redirect_stderr
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import reduce, wraps
from inspect import signature, Parameter
from io import StringIO
from pydantic import BaseModel
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

def prediction_endpoint(tag: str) -> Callable[
    [Callable[P, R]],
    Callable[[CreatePredictionInput], RemotePrediction | Iterator[RemotePrediction]]
]:
    """
    Wrap a function to handle serving remote prediction requests.

    Parameters:
        tag (str): Predictor tag.
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
        @wraps(func)
        def wrapper(input: CreatePredictionInput) -> RemotePrediction | Iterator[RemotePrediction]:
            prediction_id = _create_prediction_id()
            stdout_buffer = StringIO()
            start_time = perf_counter()
            token = _prediction_request.set(input)
            try:
                # Check args
                missing_args = required_params - set(input.inputs.keys())
                if missing_args:
                    arg_name = next(iter(missing_args))
                    raise ValueError(
                        f"Failed to create prediction because required "
                        f"input argument `{arg_name}` was not provided."
                    )
                # Deserialize inputs
                kwargs = {
                    name: _parse_remote_value(value)
                    for name, value in input.inputs.items()
                }
                # Invoke function
                with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
                    result = func(**kwargs)
                # Create prediction
                created = datetime.now(timezone.utc).isoformat()
                if input.stream:
                    return map(
                        lambda r: _create_prediction(
                            id=prediction_id,
                            tag=tag,
                            results=r,
                            start_time=start_time,
                            logs=stdout_buffer,
                            created=created
                        ),
                        result if isinstance(result, Iterator) else iter([result])
                    )
                else:
                    result = (
                        reduce(lambda _, x: x, result)
                        if isinstance(result, Iterator)
                        else result
                    )
                    return _create_prediction(
                        id=prediction_id,
                        tag=tag,
                        results=result,
                        start_time=start_time,
                        logs=stdout_buffer,
                        created=created
                    )
            except Exception:
                latency = (perf_counter() - start_time) * 1000  # millis
                return RemotePrediction(
                    id=prediction_id,
                    tag=tag,
                    latency=latency,
                    logs=stdout_buffer.getvalue(),
                    error=format_exc(),
                    created=datetime.now(timezone.utc).isoformat()
                )
            finally:
                _prediction_request.reset(token)
        # Return
        return wrapper
    return decorator

def get_prediction_request() -> CreatePredictionInput | None:
    """
    Get the current prediction request, or None if not in scope.
    """
    return _prediction_request.get()

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
    api_url: str
    access_key: str
    tag: str
    inputs: dict[str, RemoteValue]
    acceleration: RemoteAcceleration | str
    stream: bool = False