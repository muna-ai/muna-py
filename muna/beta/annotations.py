# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from typing_extensions import deprecated

from ..types import Dtype, Parameter

class Annotations:
    """
    Predictor parameter annotations for inference client compatibility.
    """

    @classmethod
    def AudioSpeed(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Audio speed parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.audio.speech.speed",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def AudioVoice(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Audio voice parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.audio.speech.voice",
            **kwargs
        )

    @classmethod
    def ChatAudios(
        cls,
        *,
        description: str,
        sample_rate: int,
        **kwargs
    ) -> Parameter:
        """
        Decoded PCM audio referenced by `{"type": "audio"}` content parts,
        in order of appearance across the conversation. All entries share
        the declared `sample_rate`.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.audios",
            sample_rate=sample_rate,
            **kwargs
        )

    @classmethod
    def ChatImages(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Decoded images referenced by `{"type": "image"}` content parts,
        in order of appearance across the conversation.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.images",
            **kwargs
        )

    @classmethod
    def ChatTools(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Tool definitions the model may call, as OpenAI function tool
        dictionaries. Rendered into the prompt by the chat template.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.tools",
            **kwargs
        )

    @classmethod
    def EmbeddingDims(
        cls,
        *,
        description: str,
        min: int | None=None,
        max: int | None=None,
        **kwargs
    ) -> Parameter:
        """
        Embedding Matryoshka dimensions parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.embeddings.dims",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def FrequencyPenalty(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Frequency penalty parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.frequency_penalty",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def MaxOutputTokens(
        cls,
        *,
        description: str,
        min: int | None=None,
        max: int | None=None,
        **kwargs
    ) -> Parameter:
        """
        Maximum output tokens parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.max_output_tokens",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def PresencePenalty(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Presence penalty parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.presence_penalty",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def ReasoningEffort(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Reasoning effort parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.reasoning_effort",
            **kwargs
        )
    
    @classmethod
    def ResponseFormat(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Response format parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.response_format",
            **kwargs
        )

    @classmethod
    @deprecated("Use `Annotations.TopP` instead.")
    def SamplingProbability(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Sampling probability parameter.

        Deprecated: Use `Annotations.TopP` instead.
        """
        return cls.TopP(
            description=description,
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    @deprecated("Use `Annotations.Temperature` instead.")
    def SamplingTemperature(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Sampling temperature parameter.

        Deprecated: Use `Annotations.Temperature` instead.
        """
        return cls.Temperature(
            description=description,
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def StopSequences(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Stop sequences parameter for the Anthropic messages API.
        """
        return Parameter(
            name="",
            description=description,
            denotation="anthropic.messages.stop_sequences",
            **kwargs
        )

    @classmethod
    def Temperature(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Sampling temperature parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.temperature",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def TopK(
        cls,
        *,
        description: str,
        min: int | None=None,
        max: int | None=None,
        **kwargs
    ) -> Parameter:
        """
        Top-K sampling parameter for the Anthropic messages API.
        """
        return Parameter(
            name="",
            description=description,
            denotation="anthropic.messages.top_k",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def TopP(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Top-P (nucleus) sampling parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.top_p",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def TranscriptionLanguage(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Transcription language parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.audio.transcriptions.language",
            **kwargs
        )

    @classmethod
    def TranscriptionPrompt(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Transcription prompt parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.audio.transcriptions.prompt",
            **kwargs
        )

    @classmethod
    def ImageCount(
        cls,
        *,
        description: str,
        min: int | None=None,
        max: int | None=None,
        **kwargs
    ) -> Parameter:
        """
        Image count parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.images.count",
            min=min,
            max=max,
            **kwargs
        )
    
    @classmethod
    def ImageWidth(
        cls,
        *,
        description: str,
        min: int | None=None,
        max: int | None=None,
        **kwargs
    ) -> Parameter:
        """
        Image width parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.images.width",
            min=min,
            max=max,
            **kwargs
        )

    @classmethod
    def ImageHeight(
        cls,
        *,
        description: str,
        min: int | None=None,
        max: int | None=None,
        **kwargs
    ) -> Parameter:
        """
        Image height parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.images.height",
            min=min,
            max=max,
            **kwargs
        )

def get_parameter(
    parameters: list[Parameter],
    *,
    dtype: Dtype | set[Dtype],
    denotation: str | None=None
) -> tuple[int | None, Parameter | None]:
    """
    Get a parameter with the given data type and denotation.
    """
    dtype = dtype if isinstance(dtype, set) else { dtype }
    for idx, param in enumerate(parameters):
        if (
            param.dtype in dtype and
            (not denotation or param.denotation == denotation)
        ):
            return idx, param
    return None, None