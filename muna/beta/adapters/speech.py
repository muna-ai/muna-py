#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from numpy import ascontiguousarray, ndarray
from pydantic import BaseModel, ConfigDict, Field
from requests import Response
from typing import Literal

from ...c import Value as NativeValue
from ...types import Dtype, Parameter, Prediction, Signature, Value
from ._adapter import Adapter
from ._utils import decode, get_parameter, result, validate_inputs, FLOATS

SpeechResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
SpeechStreamFormat = Literal["audio", "sse"]

class SpeechCreateResponse(BaseModel, **ConfigDict(arbitrary_types_allowed=True)):
    content: bytes
    response: Response

class OpenAISpeechOptions(BaseModel, **ConfigDict(extra="forbid", frozen=True, strict=True)):
    """
    Speech encoding options. SSE audio streaming is not yet supported.
    """
    response_format: SpeechResponseFormat = "mp3"
    stream_format: Literal["audio"] = "audio"

class OpenAISpeechAdapter(Adapter[Prediction, OpenAISpeechOptions, SpeechCreateResponse]):
    """
    Prepare speech inputs and encode native or remote PCM predictions.
    """
    kind: Literal["openai.audio.speech.create"] = Field("openai.audio.speech.create", init=False)
    input: Parameter
    voice: Parameter
    output: int
    sample_rate: int
    speed: Parameter | None = None

    @classmethod
    def from_signature(cls, signature: Signature) -> OpenAISpeechAdapter | None:
        """
        Create an OpenAI speech adapter from a predictor signature.
        """
        signature = signature.model_copy(deep=True)
        required = [param for param in signature.inputs if not param.optional]
        # Voice input
        _, voice = get_parameter(
            required,
            dtype=Dtype.string,
            denotation="openai.audio.speech.voice"
        )
        if voice is None:
            return None
        # Text input. Voice is also a string, so exclude it from this lookup.
        _, input = get_parameter(
            [param for param in required if param.name != voice.name],
            dtype=Dtype.string
        )
        if input is None:
            return None
        # Playback speed
        _, speed = get_parameter(
            signature.inputs,
            dtype=FLOATS,
            denotation="openai.audio.speech.speed"
        )
        if not validate_inputs(signature, [input, voice, speed]):
            return None
        # Audio output
        output, audio = get_parameter(
            signature.outputs,
            dtype=Dtype.float32,
            denotation="audio"
        )
        if audio is None:
            return None
        if audio.sample_rate is None or audio.sample_rate <= 0:
            return None
        # Return
        return cls(
            input=input,
            voice=voice,
            speed=speed,
            output=output,
            sample_rate=audio.sample_rate
        )

    def inputs(
        self,
        *,
        input: str,
        voice: str,
        speed: float = 1.
    ) -> dict[str, Value]:
        inputs = { self.input.name: input, self.voice.name: voice }
        if self.speed is not None:
            inputs[self.speed.name] = speed
        return inputs

    def adapt(
        self,
        prediction: Prediction,
        options: OpenAISpeechOptions | None = None
    ) -> SpeechCreateResponse:
        options = options or OpenAISpeechOptions()
        audio = decode(result(prediction, self.output), Dtype.float32)
        if not isinstance(audio, ndarray) or audio.dtype != "float32" or audio.ndim not in (1, 2):
            raise ValueError("Speech output must be float32 PCM shaped [frames] or [frames, channels]")
        if any(size <= 0 for size in audio.shape):
            raise ValueError("Speech output must contain audio samples")
        # Native serialization consumes a dense buffer without NumPy strides.
        content, content_type = _create_response_data(
            ascontiguousarray(audio),
            sample_rate=self.sample_rate,
            response_format=options.response_format
        )
        response = Response()
        response.status_code = 200
        response.headers["Content-Type"] = content_type
        response.headers["Content-Length"] = str(len(content))
        response._content = content
        return SpeechCreateResponse(content=content, response=response)

def _create_response_data(
    audio: ndarray,
    *,
    sample_rate: int,
    response_format: SpeechResponseFormat
) -> tuple[bytes, str]:
    channels = audio.shape[1] if audio.ndim == 2 else 1 # assume interleaved
    if response_format == "pcm":
        content_type = ";".join([
            f"audio/pcm",
            f"rate={sample_rate}",
            f"channels={channels}",
            f"encoding=float",
            f"bits=32"
        ])
        data = audio.tobytes()
        return data, content_type
    with NativeValue.from_object(audio) as audio_value:
        content_type = f"audio/{response_format};rate={sample_rate}"
        data = audio_value.serialize(content_type)
        return data, content_type
