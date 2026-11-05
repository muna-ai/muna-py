# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Annotated, Literal

DeploymentAccess = Literal["public", "unlisted", "private"]
DeploymentPricingKind = Literal["tokens", "images", "duration"]

class TokenDeploymentPricing(BaseModel):
    """
    Token-based deployment pricing.
    """
    kind: Literal["tokens"] = Field("tokens", init=False)
    currency: Literal["USD"] = Field("USD", description="Pricing currency.")
    input_per_million: float = Field(
        description="Price per million input tokens.",
        ge=0,
        serialization_alias="inputPerMillion"
    )
    cached_input_per_million: float | None = Field(
        None,
        description="Price per million cached input tokens (cache reads).",
        ge=0,
        serialization_alias="cachedInputPerMillion"
    )
    output_per_million: float | None = Field(
        None,
        description="Price per million output tokens.",
        ge=0,
        serialization_alias="outputPerMillion"
    )

class ImageDeploymentPricing(BaseModel):
    """
    Image-based deployment pricing.
    """
    kind: Literal["images"] = Field("images", init=False)
    currency: Literal["USD"] = Field("USD", description="Pricing currency.")
    per_image: float = Field(
        description="Price per image.",
        ge=0,
        serialization_alias="perImage"
    )

class DurationDeploymentPricing(BaseModel):
    """
    Duration-based deployment pricing.
    """
    kind: Literal["duration"] = Field("duration", init=False)
    currency: Literal["USD"] = Field("USD", description="Pricing currency.")
    per_minute: float = Field(
        description="Price per minute.",
        ge=0,
        serialization_alias="perMinute"
    )

# Superuser only
DeploymentPricing = Annotated[
    TokenDeploymentPricing      |
    ImageDeploymentPricing      |
    DurationDeploymentPricing,
    Field(discriminator="kind")
]

class Deployment(BaseModel):
    """
    Predictor deployment on Muna inference endpoints.

    Members:
        id (str): Deployment identifier.
        tag (str): Predictor tag.
        access (DeploymentAccess): Deployment access mode.
        created (str): Date created.
    """
    id: str = Field(description="Deployment identifier.")
    tag: str = Field(description="Predictor tag.")
    access: DeploymentAccess = Field(description="Deployment access mode.")
    created: str = Field(description="Date created.")
