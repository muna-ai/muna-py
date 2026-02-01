# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable
from typing import BinaryIO

from ...services import PredictorService, PredictionService
from ...types import Acceleration
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService
from .schema import Transcription

TranscriptionDelegate = Callable[..., object]

class TranscriptionService: # INCOMPLETE
    """
    Transcription service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__remote_predictions = remote_predictions
        self.__cache = dict[str, TranscriptionDelegate]()

    def create(
        self,
        *,
        file: BinaryIO,
        model: str,
        language: str | None=None,
        prompt: str | None=None,
        stream: bool=False,
        temperature: float=0.,
        acceleration: Acceleration | RemoteAcceleration="remote_auto"
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
            acceleration (Acceleration | RemoteAcceleration): Prediction acceleration.

        Returns:
            Transcription: Result transcription.
        """
        pass