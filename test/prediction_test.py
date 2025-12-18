# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from muna import Muna, MunaAPIError
import pytest

def test_create_raw_prediction():
    muna = Muna()
    prediction = muna.predictions.create(tag="@fxn/greeting")
    assert prediction is not None
    assert prediction.configuration is not None
    assert prediction.resources is None

def test_create_prediction():
    muna = Muna()
    radius = 4
    prediction = muna.predictions.create(
        tag="@yusuf/area",
        inputs={ "radius": radius }
    )
    assert prediction.results
    assert isinstance(prediction.results[0], float)

def test_stream_prediction():
    muna = Muna()
    stream = muna.predictions.stream(
        tag="@yusuf/generator",
        inputs={ "sentence": "The fat cat sat on the mat." }
    )
    for prediction in stream:
        assert prediction.results
        assert isinstance(prediction.results[0], str)

def test_create_remote_prediction():
    muna = Muna()
    prediction = muna.beta.predictions.remote.create(
        tag="@fxn/greeting",
        inputs={ "name": "Yusuf" }
    )
    assert prediction.results
    assert isinstance(prediction.results[0], str)

def test_stream_remote_prediction():
    muna = Muna()
    stream = muna.beta.predictions.remote.stream(
        tag="@yusuf/generator",
        inputs={ "sentence": "The fat cat sat on the mat." }
    )
    for prediction in stream:
        assert prediction.results
        assert isinstance(prediction.results[0], str)

def test_create_invalid_prediction():
    muna = Muna()
    with pytest.raises(MunaAPIError):
        muna.predictions.create(tag="@yusu/invalid-predictor")