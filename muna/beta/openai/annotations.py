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
    def SamplingTemperature(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Sampling temperature parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="openai.chat.completions.temperature",
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