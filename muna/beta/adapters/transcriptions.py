#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from numpy import array, ascontiguousarray, float32, isfinite as samples_are_finite, ndarray
from pydantic import BaseModel, ConfigDict, Field
from typing import BinaryIO, Literal

from ...c import Value as NativeValue
from ...types import Dtype, Parameter, Prediction, RemoteValue, Signature, Value
from ..types import Audio
from ._adapter import Adapter
from ._utils import decode, get_parameter, result, validate_inputs, FLOATS

class Transcription(BaseModel):
    class TokenUsage(BaseModel):
        class InputTokenDetails(BaseModel):
            audio_tokens: int
            text_tokens: int
        type: Literal["tokens"]
        input_tokens: int
        output_tokens: int
        total_tokens: int
        input_token_details: InputTokenDetails
    class DurationUsage(BaseModel):
        type: Literal["duration"]
        seconds: float
    text: str
    usage: TokenUsage | DurationUsage

class OpenAITranscriptionsOptions(BaseModel, **ConfigDict(extra="forbid", frozen=True, strict=True)):
    """
    Input audio duration used when the predictor does not return usage.
    """
    duration: float | None = Field(default=None, ge=0, allow_inf_nan=False)

class OpenAITranscriptionsAdapter(Adapter[Prediction, OpenAITranscriptionsOptions, Transcription]):
    """
    Prepare transcription PCM and adapt text with token or audio-duration usage.
    """
    kind: Literal["openai.audio.transcriptions.create"] = Field("openai.audio.transcriptions.create", init=False)
    audio: Parameter
    output: int
    usage: int | None = None
    language: Parameter | None = None
    prompt: Parameter | None = None
    temperature: Parameter | None = None

    @classmethod
    def from_signature(cls, signature: Signature) -> OpenAITranscriptionsAdapter | None:
        """
        Create an OpenAI transcriptions adapter from a predictor signature.
        """
        signature = signature.model_copy(deep=True)
        # Audio input
        _, audio = get_parameter(
            [param for param in signature.inputs if not param.optional],
            dtype={Dtype.float32, Dtype.array_list},
            denotation="audio"
        )
        if audio is None:
            return None
        if audio.sample_rate is None or audio.sample_rate <= 0:
            return None
        # Language
        _, language = get_parameter(
            signature.inputs,
            dtype=Dtype.string,
            denotation="openai.audio.transcriptions.language"
        )
        # Prompt
        _, prompt = get_parameter(
            signature.inputs,
            dtype=Dtype.string,
            denotation="openai.audio.transcriptions.prompt"
        )
        # Temperature
        _, temperature = get_parameter(
            signature.inputs,
            dtype=FLOATS,
            denotation="openai.chat.completions.temperature"
        )
        if not validate_inputs(signature, [audio, language, prompt, temperature]):
            return None
        # Text output
        output, text = get_parameter(signature.outputs, dtype={Dtype.string, Dtype.list})
        if text is None:
            return None
        # Usage is optional; returned usage takes precedence over input duration.
        _, usage = get_parameter(
            [param for param in signature.outputs if (param.value_schema or {}).get("title") == "Usage"],
            dtype=Dtype.dict
        )
        return cls(
            audio=audio,
            output=output,
            language=language,
            prompt=prompt,
            temperature=temperature,
            usage=signature.outputs.index(usage) if usage is not None else None
        )

    def inputs(
        self,
        *,
        file: BinaryIO | Audio,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float = 0.
    ) -> dict[str, Value]:
        samples = _read_audio_samples(file, sample_rate=self.audio.sample_rate)
        inputs = { self.audio.name: [samples] if self.audio.dtype == Dtype.array_list else samples }
        if self.language is not None and language is not None:
            inputs[self.language.name] = language
        if self.prompt is not None and prompt is not None:
            inputs[self.prompt.name] = prompt
        if self.temperature is not None:
            inputs[self.temperature.name] = array(temperature, dtype=self.temperature.dtype)
        return inputs

    def duration(self, inputs: dict[str, Value]) -> float:
        """
        Duration of prepared PCM, independent of prediction execution latency.
        """
        samples = inputs[self.audio.name]
        if self.audio.dtype == Dtype.array_list:
            samples = samples[0]
        return samples.shape[0] / self.audio.sample_rate

    def adapt(
        self,
        prediction: Prediction,
        options: OpenAITranscriptionsOptions | None = None
    ) -> Transcription:
        options = options or OpenAITranscriptionsOptions()
        text = result(prediction, self.output)
        if isinstance(text, RemoteValue):
            if text.dtype not in (Dtype.string, Dtype.list):
                raise ValueError("Transcription output must be a string or list")
            text = decode(text, text.dtype)
        if isinstance(text, list):
            if not text:
                raise ValueError("Transcription output must not be an empty list")
            text = text[0]
        if not isinstance(text, str):
            raise ValueError("Transcription output must be a string")
        if self.usage is not None:
            usage = decode(result(prediction, self.usage), Dtype.dict)
        elif options.duration is not None:
            usage = Transcription.DurationUsage(type="duration", seconds=options.duration)
        else:
            raise ValueError("Transcription requires returned usage or the input audio duration")
        return Transcription(text=text, usage=usage)

def _read_audio_samples(
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
                with NativeValue.from_bytes(data, f"audio/*;rate={sample_rate}") as audio_value:
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
