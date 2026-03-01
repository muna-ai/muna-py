# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable
from typing import BinaryIO

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
from ..annotations import get_parameter
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService
from ..types import Audio
from .schema import Transcription

TranscriptionDelegate = Callable[..., Transcription]

class TranscriptionService:
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
        file: BinaryIO | Audio,
        model: str,
        language: str | None=None,
        prompt: str | None=None,
        stream: bool=False,
        temperature: float=0.,
        acceleration: Acceleration | RemoteAcceleration="local_auto"
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
        # Ensure we have a delegate
        if model not in self.__cache:
            self.__cache[model] = self.__create_delegate(model)
        # Make prediction
        delegate = self.__cache[model]
        result = delegate(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            stream=stream,
            temperature=temperature,
            acceleration=acceleration
        )
        # Return
        return result

    def __create_delegate(self, tag: str) -> TranscriptionDelegate: # INCOMPLETE
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        if not predictor:
            raise ValueError(
                f"{tag} cannot be used with OpenAI transcription API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        # Check that there is only one required input parameter
        signature = predictor.signature
        required_inputs = [param for param in signature.inputs if not param.optional]
        if len(required_inputs) != 1:
            raise ValueError(
                f"{tag} cannot be used with OpenAI transcription API because "
                "it has more than one required input parameter."
            )
        # Check that the input parameter is `float32` with `audio` denotation
        _, audio_param = get_parameter(required_inputs, dtype=Dtype.float32, denotation="audio")
        if audio_param is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI transcription API because "
                "it does not have a valid audio input parameter."
            )
        # Get optional input params
        _, language_param = get_parameter(
            signature.inputs,
            dtype=Dtype.string,
            denotation="openai.audio.transcriptions.language"
        )
        _, prompt_param = get_parameter(
            signature.inputs,
            dtype=Dtype.string,
            denotation="openai.audio.transcriptions.prompt"
        )
        _, temperature_param = get_parameter(
            signature.inputs,
            dtype={ Dtype.float32, Dtype.float64 },
            denotation="openai.chat.completions.temperature"
        )
        # Define delegate
        def delegate(
            *,
            file: BinaryIO | Audio,
            model: str,
            language: str | None,
            prompt: str | None,
            stream: bool,
            temperature: float,
            acceleration: Acceleration | RemoteAcceleration
        ) -> Transcription:
            pass
        # Return
        return delegate