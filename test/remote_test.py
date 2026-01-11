# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna.beta.remote import get_prediction_request, prediction_endpoint, RemotePrediction
from muna.beta.remote.remote import _create_remote_value
from muna.beta.remote.endpoint import CreatePredictionInput
from muna.types import Dtype
from typing import Iterator

def test_prediction_endpoint_create_eager():
    payload = CreatePredictionInput(
        api_url="https://api.muna.ai/v1",
        access_key="",
        tag="@muna/greeting",
        inputs={
            "name": _create_remote_value("Yusuf"),
            "age": _create_remote_value(67)
        },
        acceleration="remote_cpu",
        stream=False
    )
    prediction = _prediction_endpoint_eager(payload)
    assert isinstance(prediction, RemotePrediction)
    assert isinstance(prediction.results, list)
    assert prediction.results[0].type == Dtype.string

def test_prediction_endpoint_stream_eager():
    @prediction_endpoint(tag="@muna/greeting")
    def predict(name: str, age: int) -> str:
        return f"Hello {name}! You are {age} years old"
    payload = CreatePredictionInput(
        api_url="https://api.muna.ai/v1",
        access_key="",
        tag="@muna/greeting",
        inputs={
            "name": _create_remote_value("Yusuf"),
            "age": _create_remote_value(67)
        },
        acceleration="remote_cpu",
        stream=True
    )
    stream = _prediction_endpoint_eager(payload)
    assert isinstance(stream, Iterator)
    assert len(list(stream)) == 1

def test_prediction_endpoint_create_generator():
    payload = CreatePredictionInput(
        api_url="https://api.muna.ai/v1",
        access_key="",
        tag="@muna/greeting",
        inputs={
            "name": _create_remote_value("Yusuf"),
            "age": _create_remote_value(67)
        },
        acceleration="remote_cpu",
        stream=False
    )
    prediction = _prediction_endpoint_generator(payload)
    assert isinstance(prediction, RemotePrediction)
    assert isinstance(prediction.results, list)
    assert prediction.results[0].type == Dtype.string

def test_prediction_endpoint_stream_generator():
    payload = CreatePredictionInput(
        api_url="https://api.muna.ai/v1",
        access_key="",
        tag="@muna/greeting",
        inputs={
            "name": _create_remote_value("Yusuf"),
            "age": _create_remote_value(67)
        },
        acceleration="remote_cpu",
        stream=True
    )
    stream = _prediction_endpoint_generator(payload)
    assert isinstance(stream, Iterator)
    assert len(list(stream)) == 2

def test_prediction_endpoint_create_missing_input():
    payload = CreatePredictionInput(
        api_url="https://api.muna.ai/v1",
        access_key="",
        tag="@muna/greeting",
        inputs={ "name": _create_remote_value("Yusuf") },
        acceleration="remote_cpu",
        stream=False
    )
    prediction = _prediction_endpoint_eager(payload)
    assert isinstance(prediction, RemotePrediction)
    assert prediction.error is not None

def test_prediction_request_input():
    request = None
    @prediction_endpoint(tag="")
    def predict(name: str, age: int):
        nonlocal request
        request = get_prediction_request()
    payload = CreatePredictionInput(
        api_url="https://api.muna.ai/v1",
        access_key="",
        tag="",
        inputs={
            "name": _create_remote_value("Yusuf"),
            "age": _create_remote_value(67)
        },
        acceleration="remote_cpu",
        stream=False
    )
    predict(payload)
    assert request is not None
    assert get_prediction_request() is None

@prediction_endpoint(tag="@muna/greeting")
def _prediction_endpoint_eager(name: str, age: int) -> str:
    return f"Hello {name}! You are {age} years old."

@prediction_endpoint(tag="@muna/greeting")
def _prediction_endpoint_generator(name: str, age: int) -> Iterator[str]:
    yield f"Hello {name}!"
    yield f"Hello {name}! You are {age} years old."