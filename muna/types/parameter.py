# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Literal

from .dtype import Dtype

Modality = Literal["audio", "embedding"]

class Parameter(BaseModel):
    """
    Predictor parameter.

    Members:
        name (str): Parameter name.
        type (Dtype): Parameter type. This is `None` if the type is unknown or unsupported by Muna.
        description (str): Parameter description.
        modality (Modality): Parameter modality for specialized data types.
        optional (bool): Whether the parameter is optional.
        range (tuple): Parameter value range for numeric parameters.
        enumeration (list): Parameter value choices for enumeration parameters.
        value_schema (dict): Parameter JSON schema. This is only populated for `list` and `dict` parameters.
        sample_rate (int): Audio sample rate in Hertz.
    """
    name: str = Field(description="Parameter name.")
    type: Dtype | None = Field(default=None, description="Parameter type. This is `None` if the type is unknown or unsupported by Muna.")
    description: str | None = Field(default=None, description="Parameter description.")
    modality: Modality | None = Field(default=None, description="Parameter modality for specialized data types.")
    optional: bool | None = Field(default=None, description="Whether the parameter is optional.")
    range: tuple[float, float] | None = Field(default=None, description="Parameter value range for numeric parameters.")
    enumeration: list[EnumerationMember] | None = Field(default=None, description="Parameter value choices for enumeration parameters.")
    value_schema: dict[str, object] | None = Field(
        default=None,
        description="Parameter JSON schema. This is only populated for `list` and `dict` parameters.",
        serialization_alias="schema",
        validation_alias=AliasChoices("schema", "value_schema")
    )
    sample_rate: int | None = Field(default=None, description="Audio sample rate in Hertz.")
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
        range: tuple[float, float]=None,
        **kwargs
    ) -> Parameter:
        """
        Numeric parameter.
        """
        return Parameter(
            name="",
            description=description,
            range=range,
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
            modality="audio",
            sample_rate=sample_rate,
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
        Embedding parameter.
        """
        return Parameter(
            name="",
            description=description,
            modality="embedding",
            **kwargs
        )

class EnumerationMember(BaseModel):
    """
    Prediction parameter enumeration member.

    Members:
        name (str): Enumeration member name.
        value (str | int): Enumeration member value.
    """
    name: str = Field(description="Enumeration member name.")
    value: str | int = Field(description="Enumeration member value.")