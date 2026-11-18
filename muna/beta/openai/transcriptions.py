#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from typing import BinaryIO

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device
from ..adapters.transcriptions import (
    OpenAITranscriptionsAdapter, OpenAITranscriptionsOptions,
    Transcription
)
from ..types import Audio

class TranscriptionService:
    """
    Transcription service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache: dict[str, OpenAITranscriptionsAdapter] = {}

    def create(
        self,
        *,
        file: BinaryIO | Audio,
        model: str,
        language: str | None=None,
        prompt: str | None=None,
        stream: bool=False,
        temperature: float=0.,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> Transcription:
        """
        Transcribe audio into the input language.

        Parameters:
            file (BinaryIO): Audio file to transcribe. MUST be flac, mp3, ogg, wav.
            model (str): Transcription model tag.
            language (str): The language of the input audio.
            prompt (str): Text to guide the model's style or continue a previous audio segment.
            stream (bool): Whether to stream transcription events.
            temperature (float): The sampling temperature, between 0 and 1.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.

        Returns:
            Transcription: Result transcription.
        """
        # Check streaming
        if stream:
            raise ValueError("Streaming transcription is not supported.")
        # Ensure adapter
        if model not in self.__cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            adapter = OpenAITranscriptionsAdapter.from_signature(predictor.signature)
            if adapter is None:
                raise ValueError(f"Predictor {model} does not support audio transcription")
            self.__cache[model] = adapter
        # Predict
        adapter = self.__cache[model]
        inputs = adapter.inputs(
            file=file,
            language=language,
            prompt=prompt,
            temperature=temperature
        )
        prediction = self.__predictions.create(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        # Adapt
        options = OpenAITranscriptionsOptions(duration=adapter.duration(inputs))
        transcription = adapter.adapt(prediction, options)
        # Return
        return transcription
