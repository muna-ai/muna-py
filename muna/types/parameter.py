# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Literal

from .dtype import Dtype

ParameterDenotation = Literal[
    "audio", "audio.speed", "audio.voice",
    "bounding_box", "depth_map",
    "embedding", "embedding.dims",
]

class EnumerationMember(BaseModel):
    """
    Parameter enumeration member.

    Members:
        name (str): Enumeration member name.
        value (str | int): Enumeration member value.
    """
    name: str = Field(description="Enumeration member name.")
    value: str | int = Field(description="Enumeration member value.")

class Parameter(BaseModel):
    """
    Predictor parameter.

    Members:
        name (str): Parameter name.
        type (Dtype): Parameter type. This is `None` if the type is unknown or unsupported by Muna.
        description (str): Parameter description.
        denotation (ParameterDenotation): Parameter denotation for specialized data types.
        optional (bool): Whether the parameter is optional.
        range (tuple): Parameter value range for numeric parameters.
        enumeration (list): Parameter value choices for enumeration parameters.
        value_schema (dict): Parameter JSON schema. This is only populated for `list` and `dict` parameters.
        sample_rate (int): Audio sample rate in Hertz.
    """
    name: str = Field(description="Parameter name.")
    type: Dtype | None = Field(default=None, description="Parameter type. This is `None` if the type is unknown or unsupported by Muna.")
    description: str | None = Field(default=None, description="Parameter description.")
    denotation: ParameterDenotation | None = Field(default=None, description="Parameter denotation for specialized data types.")
    optional: bool | None = Field(default=None, description="Whether the parameter is optional.")
    range: tuple[float, float] | None = Field(default=None, description="Parameter value range for numeric parameters.")
    enumeration: list[EnumerationMember] | None = Field(default=None, description="Parameter value choices for enumeration parameters.")
    value_schema: dict[str, object] | None = Field(
        default=None,
        description="Parameter JSON schema. This is only populated for `list` and `dict` parameters.",
        serialization_alias="schema",
        validation_alias=AliasChoices("schema", "value_schema")
    )
    sample_rate: int | None = Field(
        default=None,
        description="Audio sample rate in Hertz.",
        serialization_alias="sampleRate",
        validation_alias=AliasChoices("sample_rate", "sampleRate")
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def Generic(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Generic parameter.
        """
        return Parameter(
            name="",
            description=description,
            **kwargs
        )

    @classmethod
    def Numeric(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        **kwargs
    ) -> Parameter:
        """
        Numeric parameter.
        """
        return Parameter(
            name="",
            description=description,
            range=(min, max) if min is not None and max is not None else None,
            **kwargs
        )

    @classmethod
    def Audio(
        cls,
        *,
        description: str,
        sample_rate: int,
        **kwargs
    ) -> Parameter:
        """
        Audio parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="audio",
            sample_rate=sample_rate,
            **kwargs
        )

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
            denotation="audio.speed",
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
            denotation="audio.voice",
            **kwargs
        )

    @classmethod
    def Embedding(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Embedding matrix parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="embedding",
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
            denotation="embedding.dims",
            range=(min, max) if min is not None and max is not None else None,
            **kwargs
        )

    @classmethod
    def BoundingBox(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Bounding box parameter.
        NOTE: The box MUST be specified in normalized coordinates.
        """
        return Parameter(
            name="",
            description=description,
            denotation="bounding_box",
            **kwargs
        )
    
    @classmethod
    def BoundingBoxes(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Bounding box collection parameter.
        NOTE: The boxes MUST be specified in normalized coordinates.
        """
        return Parameter.BoundingBox(
            description=description,
            **kwargs
        )
    
    @classmethod
    def DepthMap(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Depth map parameter.
        """
        return Parameter(
            name="",
            description=description,
            denotation="depth_map",
            **kwargs
        )