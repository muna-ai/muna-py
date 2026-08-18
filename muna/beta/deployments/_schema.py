# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Annotated, Literal

DeploymentKind = Literal["shared", "dedicated"]
DeploymentGPU = Literal["a100", "h100", "h200", "b200"]
DeploymentPricingKind = Literal["tokens", "images", "duration"]
DeploymentProvider = Literal["baremetal", "baseten",  "modal", "spheron"]

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

DeploymentPricing = Annotated[
    TokenDeploymentPricing      |
    ImageDeploymentPricing      |
    DurationDeploymentPricing,
    Field(discriminator="kind")
]

class DeploymentSpec(BaseModel):
    """
    Predictor deployment specification.
    """
    tag: str = Field(description="Predictor tag.")
    kind: DeploymentKind = Field(description="Deployment kind.")
    provider: DeploymentProvider = Field(description="Deployment compute provider.")
    name: str = Field(description="Human readable deployment name.")
    cpu: int | None = Field(None, description="Deployment vCPU count.")
    gpu: DeploymentGPU | None = Field(None, description="Deployment GPU.")
    gpu_count: int | None = Field(None, description="Deployment GPU count.", ge=1)
    memory: int | None = Field(None, description="Deployment memory size in megabytes.")
    concurrency: int | None = Field(None, description="Deployment request concurrency for load balancing.")
    min_replicas: int | None = Field(None, description="Deployment minimum replica count for autoscaling.")
    max_replicas: int | None = Field(None, description="Deployment maximum replica count for autoscaling.")
    scaledown_window: float | None = Field(None, description="Deployment scaledown window in seconds for autoscaling.")
    ssh_host: str | None = Field(None, description="Deployment SSH host URL. Required for baremetal deployments.")
    endpoint_url: str | None = Field(None, description="Deployment URL override. Required for baremetal deployments.")
    pricing: DeploymentPricing | None = Field(None, description="Deployment pricing.")

class Deployment(BaseModel, ABC):
    endpoint_url: str = Field(description="Deployment endpoint URL.")
    dashboard_url: str | None = Field(None, description="Deployment management URL on provider platform.")

    @abstractmethod
    def wait(self):
        """
        Block until the deployment is live and healthy.
        """