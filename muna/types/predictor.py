# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

from .parameter import Parameter
from .user import User

PredictorAccess = Literal["public", "private", "unlisted"]

PredictorStatus = Literal["compiling", "active", "archived"]

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

class Signature(BaseModel):
    """
    Predictor signature.

    Members:
        inputs (list): Input parameters.
        outputs (list): Output parameters.
    """
    inputs: list[Parameter] = Field(description="Input parameters.")
    outputs: list[Parameter] = Field(description="Output parameters.")