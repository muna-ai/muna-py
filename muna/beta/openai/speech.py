#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device
from ..adapters.speech import (
    OpenAISpeechAdapter, OpenAISpeechOptions, SpeechCreateResponse,
    SpeechResponseFormat, SpeechStreamFormat
)

class SpeechService:
    """
    Speech service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache: dict[str, OpenAISpeechAdapter] = {}

    def create(
        self,
        *,
        input: str,
        model: str,
        voice: str,
        response_format: SpeechResponseFormat="mp3",
        speed: float=1.,
        stream_format: SpeechStreamFormat="audio",
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> SpeechCreateResponse:
        """
        Generate audio from the input text.

        Parameters:
            input (str): The text to generate audio for.
            model (str): Speech generation model tag.
            voice (str): Voice to use when generating the audio.
            response_format ("mp3" | "opus" | "aac" | "flac" | "wav" | "pcm"): Audio output format.
            speed (float): The speed of the generated audio. Defaults to 1.0.
            stream_format ("audio" | "sse"):  The format to stream the audio in.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.
        """
        # Ensure adapter
        if model not in self.__cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            self.__cache[model] = OpenAISpeechAdapter.from_signature(predictor.signature)
        # Predict
        adapter = self.__cache[model]
        inputs = adapter.inputs(
            input=input,
            voice=voice,
            speed=speed
        )
        prediction = self.__predictions.create(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        # Adapt
        options = OpenAISpeechOptions(
            response_format=response_format,
            stream_format=stream_format
        )
        response = adapter.adapt(prediction, options)
        # Return
        return response
