# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing import Any, Literal

from .dtype import Dtype
from .user import User

PredictorAccess = Literal["public", "private", "unlisted"]

PredictorStatus = Literal["compiling", "active", "archived"]

class EnumerationMember(BaseModel):
    """
    Prediction parameter enumeration member.

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
        optional (bool): Whether the parameter is optional.
        range (tuple): Parameter value range for numeric parameters.
        enumeration (list): Parameter value choices for enumeration parameters.
        value_schema (dict): Parameter JSON schema. This is only populated for `list` and `dict` parameters.
    """
    name: str = Field(description="Parameter name.")
    type: Dtype | None = Field(default=None, description="Parameter type. This is `None` if the type is unknown or unsupported by Muna.")
    description: str | None = Field(default=None, description="Parameter description.")
    optional: bool | None = Field(default=None, description="Whether the parameter is optional.")
    range: tuple[float | None, float | None] | None = Field(default=None, description="Parameter value range for numeric parameters.")
    enumeration: list[EnumerationMember] | None = Field(default=None, description="Parameter value choices for enumeration parameters.")
    value_schema: dict[str, Any] | None = Field(default=None, description="Parameter JSON schema. This is only populated for `list` and `dict` parameters.", serialization_alias="schema", validation_alias=AliasChoices("schema", "value_schema"))
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Signature(BaseModel):
    """
    Predictor signature.

    Members:
        inputs (list): Input parameters.
        outputs (list): Output parameters.
    """
    inputs: list[Parameter] = Field(description="Input parameters.")
    outputs: list[Parameter] = Field(description="Output parameters.")

class Predictor(BaseModel):
    """
    Predictor.

    Members:
        tag (str): Predictor tag.
        owner (User): Predictor owner.
        name (str): Predictor name.
        status (PredictorStatus): Predictor status.
        access (PredictorAccess): Predictor access.
        signature (Signature): Predictor signature.
        created (str): Date created.
        description (str): Predictor description.
        card (str): Predictor card.
        media (str): Predictor media URL.
        license (str): Predictor license URL.
    """
    tag: str = Field(description="Predictor tag.")
    owner: User = Field(description="Predictor owner.")
    name: str = Field(description="Predictor name.")
    status: PredictorStatus = Field(description="Predictor status.")
    access: PredictorAccess = Field(description="Predictor access.")
    signature: Signature = Field(description="Predictor signature.")
    created: str = Field(description="Date created.")
    description: str | None = Field(default=None, description="Predictor description.")
    card: str | None = Field(default=None, description="Predictor card.")
    media: str | None = Field(default=None, description="Predictor media URL.")
    license: str | None = Field(default=None, description="Predictor license URL.")