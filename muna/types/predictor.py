# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

from .parameter import Parameter
from .user import User

PredictorAccess = Literal["public", "private", "unlisted"]
PredictorDistribution = Literal["hosted", "community"]
PredictorStatus = Literal["compiling", "active", "archived"]

class Predictor(BaseModel):
    """
    Predictor.

    Members:
        tag (str): Predictor tag.
        owner (User): Predictor owner.
        name (str): Predictor name.
        description (str): Predictor description.
        status (PredictorStatus): Predictor status.
        access (PredictorAccess): Predictor access.
        signature (Signature): Predictor signature.
        created (str): Date created.
    """
    tag: str = Field(description="Predictor tag.")
    owner: User = Field(description="Predictor owner.")    
    name: str = Field(description="Predictor name.")
    description: str = Field(description="Predictor description.")
    status: PredictorStatus = Field(description="Predictor status.")
    access: PredictorAccess = Field(description="Predictor access.")
    signature: Signature = Field(description="Predictor signature.")
    created: str = Field(description="Date created.")

class Signature(BaseModel):
    """
    Predictor signature.

    Members:
        inputs (list): Input parameters.
        outputs (list): Output parameters.
    """
    inputs: list[Parameter] = Field(description="Input parameters.")
    outputs: list[Parameter] = Field(description="Output parameters.")