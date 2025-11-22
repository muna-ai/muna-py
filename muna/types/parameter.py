# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from numpy import ndarray
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr
from typing import Final, Literal

from .value import Dtype, RemoteValue, Value

class _NotGivenType:
    __slots__ = ()
    def __repr__(self) -> str:
        return "NOT_GIVEN"

NOT_GIVEN: Final = _NotGivenType()

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

class Parameter(BaseModel, **ConfigDict(arbitrary_types_allowed=True)):
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
    type: Dtype | None = Field(
        default=None,
        description="Parameter type. This is `None` if the type is unknown or unsupported by Muna."
    )
    description: str | None = Field(
        default=None,
        description="Parameter description."
    )
    denotation: ParameterDenotation | None = Field(
        default=None,
        description="Parameter denotation for specialized data types."
    )
    optional: bool | None = Field(
        default=None,
        description="Whether the parameter is optional."
    )
    range: tuple[float, float] | None = Field(
        default=None,
        description="Parameter value range for numeric parameters."
    )
    enumeration: list[EnumerationMember] | None = Field(
        default=None,
        description="Parameter value choices for enumeration parameters."
    )
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
    example: RemoteValue | None = Field(default=None, description="Parameter example value for testing.")
    _example: Value | _NotGivenType = PrivateAttr(default=NOT_GIVEN)

    @classmethod
    def Generic(
        cls,
        *,
        description: str,
        example: Value=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Generic parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            **kwargs
        )
        parameter._example = example
        return 

    @classmethod
    def Numeric(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        example: float | int=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Numeric parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            range=(min, max) if min is not None and max is not None else None,
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def Audio(
        cls,
        *,
        description: str,
        sample_rate: int,
        example: ndarray=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Audio parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            denotation="audio",
            sample_rate=sample_rate,
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def AudioSpeed(
        cls,
        *,
        description: str,
        min: float | None=None,
        max: float | None=None,
        example: float=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Audio speed parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            denotation="audio.speed",
            range=(min, max) if min is not None and max is not None else None,
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def AudioVoice(
        cls,
        *,
        description: str,
        example: str | int=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Audio voice parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            denotation="audio.voice",
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def Embedding(
        cls,
        *,
        description: str,
        example: ndarray=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Embedding matrix parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            denotation="embedding",
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def EmbeddingDims(
        cls,
        *,
        description: str,
        min: int | None=None,
        max: int | None=None,
        example: int=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Embedding Matryoshka dimensions parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            denotation="embedding.dims",
            range=(min, max) if min is not None and max is not None else None,
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def BoundingBox(
        cls,
        *,
        description: str,
        example: dict[str, object]=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Bounding box parameter.
        NOTE: The box MUST be specified in normalized coordinates.
        """
        parameter = Parameter(
            name="",
            description=description,
            denotation="bounding_box",
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def BoundingBoxes(
        cls,
        *,
        description: str,
        example: list[dict[str, object]]=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Bounding box collection parameter.
        NOTE: The boxes MUST be specified in normalized coordinates.
        """
        parameter = Parameter.BoundingBox(
            description=description,
            **kwargs
        )
        parameter._example = example
        return parameter

    @classmethod
    def DepthMap(
        cls,
        *,
        description: str,
        example: ndarray=NOT_GIVEN,
        **kwargs
    ) -> Parameter:
        """
        Depth map parameter.
        """
        parameter = Parameter(
            name="",
            description=description,
            denotation="depth_map",
            **kwargs
        )
        parameter._example = example
        return parameter