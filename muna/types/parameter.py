# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .value import Dtype

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
        denotation (str): Parameter denotation for specialized data types.
        optional (bool): Whether the parameter is optional.
        range (tuple): Parameter value range for numeric parameters.
        enumeration (list): Parameter value choices for enumeration parameters.
        value_schema (dict): Parameter JSON schema. This is only populated for `list` and `dict` parameters.
        sample_rate (int): Audio sample rate in Hertz.
    """
    name: str = Field(description="Parameter name.")
    dtype: Dtype | None = Field(
        default=None,
        description="Parameter type. This is `None` if the type is unknown or unsupported by Muna.",
        validation_alias=AliasChoices("dtype", "type")
    )
    description: str | None = Field(
        default=None,
        description="Parameter description."
    )
    denotation: str | None = Field(
        default=None,
        description="Parameter denotation for specialized data types."
    )
    optional: bool | None = Field(
        default=None,
        description="Whether the parameter is optional."
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
    min: int | float | None = Field(
        default=None,
        description="Parameter minumum value."
    )
    max: int | float | None = Field(
        default=None,
        description="Parameter maximum value."
    )
    sample_rate: int | None = Field(
        default=None,
        description="Audio sample rate in Hertz.",
        serialization_alias="sampleRate",
        validation_alias=AliasChoices("sample_rate", "sampleRate")
    )

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
            min=min,
            max=max,
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

    @classmethod
    def SegmentationMap(
        cls,
        *,
        description: str,
        **kwargs
    ) -> Parameter:
        """
        Segmentation map parameter.

        This must be a boolean tensor with shape (H,W) or (N,H,W).
        """
        return Parameter(
            name="",
            description=description,
            denotation="segmentation_map",
            **kwargs
        )