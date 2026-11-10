# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from datetime import timedelta
from math import isfinite
from pydantic import BaseModel, ConfigDict, Field
from re import finditer, fullmatch
from typing import Annotated, Literal

DeploymentPricingKind = Literal["tokens", "images", "duration"]
DeploymentTenancy = Literal["shared", "dedicated"]

class _CustomDeploymentPricing(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

class TokenDeploymentPricing(_CustomDeploymentPricing):
    """
    Token-based deployment pricing.
    """
    kind: Literal["tokens"] = Field("tokens", init=False)
    currency: Literal["USD"] = Field("USD", description="Pricing currency.")
    input_per_million: float = Field(
        description="Price per million input tokens.",
        ge=0,
        alias="inputPerMillion"
    )
    cached_input_per_million: float | None = Field(
        None,
        description="Price per million cached input tokens (cache reads).",
        ge=0,
        alias="cachedInputPerMillion"
    )
    output_per_million: float | None = Field(
        None,
        description="Price per million output tokens.",
        ge=0,
        alias="outputPerMillion"
    )

class ImageDeploymentPricing(_CustomDeploymentPricing):
    """
    Image-based deployment pricing.
    """
    kind: Literal["images"] = Field("images", init=False)
    currency: Literal["USD"] = Field("USD", description="Pricing currency.")
    per_image: float = Field(
        description="Price per image.",
        ge=0,
        alias="perImage"
    )

class DurationDeploymentPricing(_CustomDeploymentPricing):
    """
    Audio content duration pricing, using returned usage rather than execution latency.
    """
    kind: Literal["duration"] = Field("duration", init=False)
    currency: Literal["USD"] = Field("USD", description="Pricing currency.")
    per_minute: float = Field(
        description="Price per minute.",
        ge=0,
        alias="perMinute"
    )

class ComputeDeploymentPricing(BaseModel, **ConfigDict(populate_by_name=True)):
    """
    Current compute rate revision IDs, keyed by architecture (for example, sm_100).
    This response quote cannot be supplied as a custom pricing override.
    """
    kind: Literal["compute"] = "compute"
    currency: Literal["USD"] = "USD"
    rate_revisions: dict[str, str] = Field(alias="rateRevisions")

DeploymentPricing = Annotated[
    TokenDeploymentPricing      |
    ImageDeploymentPricing      |
    DurationDeploymentPricing   |
    ComputeDeploymentPricing,
    Field(discriminator="kind")
]

class DeploymentProgress(BaseModel):
    """
    One reduced preparation attempt; fractions can fall when retrying.
    """
    stage: Literal["preparing", "waiting", "resolving", "downloading"]
    fraction: float | None = Field(None, ge=0, le=1, allow_inf_nan=False)
    retrying: bool = False

class DeploymentPreparation(BaseModel):
    state: Literal["preparing", "ready", "failed"]
    started: str
    completed: str | None = None

class DeploymentHealth(BaseModel):
    state: Literal["unknown", "available", "degraded", "unavailable"]
    reason: str | None = None
    since: str

class DeploymentFailure(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    stage: Literal["resolve", "download", "load"]
    code: str
    message: str
    retrying: bool
    first_observed: str = Field(alias="firstObserved")
    last_observed: str = Field(alias="lastObserved")

class DeploymentStatus(BaseModel):
    preparation: DeploymentPreparation
    health: DeploymentHealth
    progress: DeploymentProgress | None = None
    failure: DeploymentFailure | None = None
    observed: str

class Deployment(BaseModel):
    """
    Predictor deployment on Muna inference endpoints.

    Members:
        id (str): Deployment identifier.
        tag (str): Predictor tag.
        listed (bool): Whether a public predictor is listed in the public catalog.
        pricing (DeploymentPricing | None): Custom prices or current compute quote.
        tenancy (DeploymentTenancy): Whether the deployment shares GPUs with other deployments.
        max_queue_time (timedelta): Maximum time a request waits for compute before it is rejected.
        created (str): Date created.
        status (DeploymentStatus | None): Initial disk preparation and ongoing health.
    """
    id: str = Field(description="Deployment identifier.")
    tag: str = Field(description="Predictor tag.")
    listed: bool = Field(False, description="Whether a public predictor is listed in the public catalog.")
    pricing: DeploymentPricing | None = Field(None, description="Custom prices or current compute quote.")
    tenancy: DeploymentTenancy = Field(
        default="shared",
        description="Whether the deployment shares GPUs with other deployments."
    )
    max_queue_time: timedelta | None = Field(
        default=None,
        description="Maximum time a request waits for compute before it is rejected.",
        validation_alias="maxQueueTime"
    )
    created: str = Field(description="Date created.")
    status: DeploymentStatus | None = Field(None, description="Initial preparation and ongoing health summary.")

def parse_max_queue_time(value: timedelta | str) -> timedelta:
    """
    Parse a positive queue time of at most 10 minutes, like '30s' or '1m30s'.
    Bare numbers are rejected so the unit is explicit.
    """
    if isinstance(value, timedelta):
        duration = value
    elif isinstance(value, str) and fullmatch(rf"(?:{_DURATION_PATTERN}\s*)+", value.strip().lower()):
        seconds = sum(
            float(amount) * _DURATION_UNITS[unit]
            for amount, unit in (m.groups() for m in finditer(_DURATION_PATTERN, value.strip().lower()))
        )
        if not isfinite(seconds) or seconds > 600:
            raise ValueError("Max queue time must not exceed 10 minutes")
        duration = timedelta(seconds=seconds)
    else:
        raise ValueError(
            f"Max queue time must be a `timedelta` or a duration string like "
            f"'30s' or '10m', not {value!r}"
        )
    if duration <= timedelta(0):
        raise ValueError(f"Max queue time must be positive, not {value!r}")
    if duration > timedelta(minutes=10):
        raise ValueError("Max queue time must not exceed 10 minutes")
    return duration

_DURATION_PATTERN = r"(\d+(?:\.\d+)?)\s*([smhd])"
_DURATION_UNITS = { "s": 1, "m": 60, "h": 3600, "d": 86400 }
