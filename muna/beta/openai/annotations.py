# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ...types.parameter import Parameter

class Annotations:
    """
    OpenAI annotations.
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
            range=(min, max) if min is not None and max is not None else None,
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
            range=(min, max) if min is not None and max is not None else None,
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
            range=(min, max) if min is not None and max is not None else None,
            **kwargs
        )

    @classmethod
    def MaxOutputTokens(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Maximum output tokens parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.max_output_tokens",
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
            range=(min, max) if min is not None and max is not None else None,
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
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.top_p",
            range=(min, max) if min is not None and max is not None else None,
            **kwargs
        )

    @classmethod
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
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.temperature",
            range=(min, max) if min is not None and max is not None else None,
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