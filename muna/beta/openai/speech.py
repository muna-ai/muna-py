# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable
from numpy import ndarray
from requests import Response
from typing import Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService
from .schema import SpeechCreateResponse

SpeechDelegate = Callable[..., object]

class SpeechService:
    """
    Speech service.
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
        self.__cache = dict[str, SpeechDelegate]()
    
    def create( # DEPLOY
        self,
        *,
        input: str,
        model: str,
        voice: str,
        response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]="mp3",
        speed: float=1.,
        stream_format: Literal["audio", "sse"]="audio",
        acceleration: Acceleration | RemoteAcceleration="remote_auto"
    ) -> SpeechCreateResponse:
        """
        Generate audio from the input text.

        Parameters:
            input (str): The text to generate audio for.
            model (str): Speech generation predictor tag.
            voice (str): Voice to use when generating the audio.
            response_format ("mp3" | "opus" | "aac" | "flac" | "wav" | "pcm"): Audio output format.
            speed (float): The speed of the generated audio. Defaults to 1.0.
            stream_format ("audio" | "sse"):  The format to stream the audio in.
            acceleration (Acceleration | RemoteAcceleration): Prediction acceleration.
        """
        # Ensure we have a delegate
        if model not in self.__cache:
            self.__cache[model] = self.__create_delegate(model)
        # Make prediction
        delegate = self.__cache[model]
        result = delegate(
            input=input,
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
            stream_format=stream_format,
            acceleration=acceleration
        )
        # Return
        return result

    def __create_delegate(self, tag: str) -> SpeechDelegate:
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        if not predictor:
            raise ValueError(
                f"{tag} cannot be used with OpenAI speech API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        # Get required inputs
        signature = predictor.signature
        required_inputs = [param for param in signature.inputs if not param.optional]
        if len(required_inputs) != 2:
            raise ValueError(
                f"{tag} cannot be used with OpenAI speech API because "
                "it does not have exactly two required input parameters."
            )
        # Get the text input param
        input_param = next((param for param in required_inputs if param.type == Dtype.string), None)
        if not input_param:
            raise ValueError(
                f"{tag} cannot be used with OpenAI speech API because "
                "it does not have the required speech input parameter."
            )
        # Get the voice input param
        voice_param = next((
            param
            for param in required_inputs
            if param.type == Dtype.string and param.denotation == "audio.voice"
        ), None)
        if not voice_param:
            raise ValueError(
                f"{tag} cannot be used with OpenAI speech API because "
                "it does not have the required speech voice parameter."
            )
        # Get the speed input param (optional)
        speed_param = next((
            param
            for param in signature.inputs
            if param.type in [Dtype.float32, Dtype.float64]
        ), None)
        # Get the audio output parameter index
        audio_param_idx = next((
            idx
            for idx, param in enumerate(signature.outputs)
            if param.type == Dtype.float32 and param.denotation == "audio"
        ), None)
        if audio_param_idx is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI speech API because "
                "it has no outputs with an `audio` denotation."
            )
        audio_param = signature.outputs[audio_param_idx]
        # Define delegate
        def delegate(
            *,
            input: str,
            model: str,
            voice: str,
            response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"],
            speed: float,
            stream_format: Literal["audio", "sse"],
            acceleration: Acceleration | RemoteAcceleration
        ) -> SpeechCreateResponse:
            # Check response format
            if response_format != "pcm":
                raise ValueError(
                    f"Cannot create speech with response format `{response_format}` "
                    f"because only `pcm` is currently supported."
                )
            # Check stream format
            if stream_format != "audio":
                raise ValueError(
                    f"Cannot create speech with stream format `{stream_format}` "
                    f"because only `audio` is currently supported."
                )
            # Get prediction creation function (local or remote)
            create_prediction_func = (
                self.__remote_predictions.create
                if acceleration.startswith("remote_")
                else self.__predictions.create
            )
            # Build prediction input map
            prediction_inputs = {
                input_param.name: input,
                voice_param.name: voice
            }
            if speed_param is not None:
                prediction_inputs[speed_param.name] = speed
            # Create prediction
            prediction = create_prediction_func(
                tag=model,
                inputs=prediction_inputs,
                acceleration=acceleration
            )
            # Check for error
            if prediction.error:
                raise RuntimeError(prediction.error)
            # Check returned audio
            audio = prediction.results[audio_param_idx]
            if not isinstance(audio, ndarray):
                raise RuntimeError(f"{tag} returned object of type {type(audio)} instead of an audio tensor")
            if audio.ndim not in [1, 2]:
                raise RuntimeError(f"{tag} returned audio tensor with invalid shape: {audio.shape}")
            # Create response
            channels = audio.shape[0] if audio.ndim == 2 else 1 # assume planar
            content = audio.tobytes()
            response = Response()
            response.status_code = 200
            response.headers = {
                "Content-Type": f"audio/pcm;rate={audio_param.sample_rate};channels={channels};encoding=float;bits=32",
                "Content-Length": len(content)
            }
            response._content = content
            result = SpeechCreateResponse(
                content=content,
                response=response
            )
            # Return
            return result
        # Return
        return delegate