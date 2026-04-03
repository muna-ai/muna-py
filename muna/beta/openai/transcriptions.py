# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable
from numpy import ndarray
from typing import BinaryIO

from ...c import Value
from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
from ..annotations import get_parameter
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
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
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
        acceleration: Acceleration="local_auto"
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

    def __create_delegate(self, tag: str) -> TranscriptionDelegate:
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
        # Check that the input parameter is `float32` or `array_list` with `audio` denotation
        _, audio_param = get_parameter(
            required_inputs,
            dtype={ Dtype.float32, Dtype.array_list },
            denotation="audio"
        )
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
        # Get the transcription output param index
        transcription_param_idx, transcription_param = get_parameter(
            signature.outputs,
            dtype={ Dtype.string, Dtype.list }
        )
        if transcription_param is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI transcription API because "
                "it has no output string parameter."
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
            acceleration: Acceleration
        ) -> Transcription:
            # Build prediction input map
            samples = self.__read_audio_samples(
                file,
                sample_rate=audio_param.sample_rate
            )
            audio_input = [samples] if audio_param.dtype == Dtype.array_list else samples
            input_map = { audio_param.name: audio_input }
            if language_param is not None and language is not None:
                input_map[language_param.name] = language
            if prompt_param is not None and prompt is not None:
                input_map[prompt_param.name] = prompt
            if temperature_param is not None:
                input_map[temperature_param.name] = temperature
            # Create prediction
            prediction = self.__predictions.create(
                tag=model,
                inputs=input_map,
                acceleration=acceleration
            )
            # Check for error
            if prediction.error:
                raise RuntimeError(prediction.error)
            # Extract transcription text
            text = prediction.results[transcription_param_idx]
            if isinstance(text, list):
                text = text[0]
            if not isinstance(text, str):
                raise RuntimeError(f"{tag} returned object of type {type(text)} instead of a string")
            # Create result
            duration = samples.shape[0] / audio_param.sample_rate
            result = Transcription(
                text=text,
                usage=Transcription.DurationUsage(
                    type="duration",
                    seconds=duration
                )
            )
            # Return
            return result
        # Return
        return delegate

    def __read_audio_samples(
        self,
        file: BinaryIO | Audio,
        *,
        sample_rate: int
    ) -> ndarray:
        match file:
            case Audio():
                if file.sample_rate != sample_rate:
                    raise ValueError(
                        f"Audio sample rate {file.sample_rate}Hz does not match "
                        f"the required sample rate of {sample_rate}Hz."
                    )
                return file.samples
            case file if hasattr(file, "read"):
                with Value.from_bytes(
                    file.read(),
                    f"audio/*;rate={sample_rate}"
                ) as audio_value:
                    return audio_value.to_object()
            case _:
                raise TypeError(
                    f"Expected `BinaryIO` or `Audio` instance "
                    f"but got `{type(file).__qualname__}`"
                )