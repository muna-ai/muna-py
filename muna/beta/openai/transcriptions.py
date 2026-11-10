# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable
from math import isfinite
from numpy import array, ascontiguousarray, float32, isfinite as samples_are_finite, ndarray
from typing import BinaryIO

from ...c import Value
from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device, Dtype
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
        if stream:
            raise ValueError("Streaming transcription is not supported.")
        if not isfinite(temperature) or not 0. <= temperature <= 1.:
            raise ValueError("Transcription temperature must be between 0 and 1.")
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
            acceleration=acceleration,
            devices=devices
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
        sample_rate = audio_param.sample_rate
        if sample_rate is None or sample_rate <= 0:
            raise ValueError("Transcription audio input must declare a positive sample rate.")
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
        if transcription_param is None or transcription_param_idx is None:
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
            acceleration: Acceleration,
            devices: list[Device | str] | None
        ) -> Transcription:
            # Build prediction input map
            samples = self.__read_audio_samples(
                file,
                sample_rate=sample_rate
            )
            audio_input = [samples] if audio_param.dtype == Dtype.array_list else samples
            input_map = { audio_param.name: audio_input }
            if language_param is not None and language is not None:
                input_map[language_param.name] = language
            if prompt_param is not None and prompt is not None:
                input_map[prompt_param.name] = prompt
            if temperature_param is not None:
                input_map[temperature_param.name] = array(temperature, dtype=temperature_param.dtype)
            # Create prediction
            prediction = self.__predictions.create(
                tag=model,
                inputs=input_map,
                acceleration=acceleration,
                devices=devices
            )
            # Check for error
            if prediction.error:
                raise RuntimeError(prediction.error)
            # Extract transcription text
            if prediction.results is None or transcription_param_idx >= len(prediction.results):
                raise RuntimeError(f"{tag} returned no transcription result")
            text = prediction.results[transcription_param_idx]
            if isinstance(text, list):
                if not text:
                    raise RuntimeError(f"{tag} returned an empty transcription list")
                text = text[0]
            if not isinstance(text, str):
                raise RuntimeError(f"{tag} returned object of type {type(text)} instead of a string")
            # Create result
            duration = samples.shape[0] / sample_rate
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
                samples = file.samples
                expected_channels = samples.shape[1] if isinstance(samples, ndarray) and samples.ndim == 2 else 1
                if file.channel_count != expected_channels:
                    raise ValueError("Audio channel count does not match the PCM shape.")
            case file if hasattr(file, "read"):
                data = file.read()
                if not data:
                    raise ValueError("Audio file is empty.")
                try:
                    with Value.from_bytes(data, f"audio/*;rate={sample_rate}") as audio_value:
                        samples = audio_value.to_object()
                except RuntimeError as error:
                    raise ValueError(f"Cannot decode audio: {error}") from error
            case _:
                raise TypeError(
                    f"Expected `BinaryIO` or `Audio` instance "
                    f"but got `{type(file).__qualname__}`"
                )
        if (
            not isinstance(samples, ndarray) or
            samples.dtype != float32 or
            samples.ndim not in (1, 2) or
            any(dimension <= 0 for dimension in samples.shape) or
            not samples_are_finite(samples).all()
        ):
            raise ValueError("Audio must be nonempty float32 PCM shaped [frames] or [frames, channels].")
        # Native Value maps a dense buffer and does not carry NumPy strides.
        return ascontiguousarray(samples)
