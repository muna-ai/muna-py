# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import cast

from ...api import MunaClient
from ...types import Acceleration
from ._schema import (
    Deployment, DeploymentGPU, DeploymentKind, DeploymentPricing,
    DeploymentProvider, DeploymentSpec
)
from .baremetal import create_baremetal_deployment
from .baseten import create_baseten_deployment
from .modal import create_modal_deployment
from .spheron import create_spheron_deployment

DeploymentAcceleration = Acceleration | tuple[Acceleration, int] 

class DeploymentService:
    """
    Deploy compiled models to compute clouds.
    """

    def __init__(self, client: MunaClient):
        self.client = client

    def create(
        self,
        tag: str,
        provider: DeploymentProvider,
        acceleration: DeploymentAcceleration,
        *,
        name: str | None = None,
        kind: DeploymentKind = "dedicated",
        cpu: int | None = None,
        memory: int | None = None,
        concurrency: int | None = None,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
        scaledown_window: float | None = None,
        ssh_host: str | None = None,
        endpoint_url: str | None = None,
        pricing: DeploymentPricing | None = None
    ) -> Deployment:
        """
        Create a deployment.

        Parameters:
            tag (str): Predictor tag.
            provider (DeploymentProvider): Compute provider.
            acceleration (DeploymentAcceleration): Accelerator, optionally paired with its count.
            name (str): Human-readable deployment name.
            kind (DeploymentKind): Deployment availability kind.
            cpu (int): Number of virtual CPUs.
            memory (int): Memory size in megabytes.
            concurrency (int): Request concurrency for load balancing.
            min_replicas (int): Minimum replica count for autoscaling.
            max_replicas (int): Maximum replica count for autoscaling.
            scaledown_window (float): Autoscaling scale-down window in seconds.
            ssh_host (str): SSH target for a baremetal deployment.
            endpoint_url (str): Public base URL for a baremetal deployment.
            pricing (DeploymentPricing): Pricing for a shared deployment.

        Returns:
            Deployment: Created deployment.
        """
        gpu, gpu_count = _parse_gpu(acceleration)
        spec = DeploymentSpec(
            tag=tag,
            provider=provider,
            kind=kind,
            name=name or f"Muna: {tag}",
            cpu=cpu,
            gpu=gpu,
            gpu_count=gpu_count,
            memory=memory,
            concurrency=concurrency,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scaledown_window=scaledown_window,
            ssh_host=ssh_host,
            endpoint_url=endpoint_url,
            pricing=pricing
        )
        record = self.__create_record(spec)
        if spec.kind == "shared":
            return _SharedDeployment(endpoint_url=record.endpoint or "")
        deployment = _create_deployment(spec, record.key)
        self.__update_record(
            record.id,
            endpoint=deployment.endpoint_url
        )
        return deployment

    def __create_record(self, spec: DeploymentSpec) -> _DeploymentRecord:
        """
        Create a deployment record.
        """
        body = _CreateDeploymentRecordInput(
            tag=spec.tag,
            name=spec.name,
            provider=spec.provider,
            kind=spec.kind,
            gpu=spec.gpu,
            gpu_count=spec.gpu_count,
            pricing=spec.pricing
        )
        return self.client.request(
            method="POST",
            path="/deployments",
            body=body,
            response_type=_DeploymentRecord
        )

    def __update_record(
        self,
        deployment_id: str,
        *,
        endpoint: str | None
    ) -> _DeploymentRecord:
        """
        Update a deployment record.
        """
        if endpoint is None:
            return
        body = _UpdateDeploymentRecordInput(
            endpoint=endpoint
        )
        return self.client.request(
            method="PATCH",
            path=f"/deployments/{deployment_id}",
            body=body,
            response_type=_DeploymentRecord
        )

def _create_deployment(
    spec: DeploymentSpec,
    key: str
) -> Deployment:
    """
    Create a deployment.
    """
    match spec.provider:
        case "baremetal": return create_baremetal_deployment(spec, deployment_key=key)
        case "baseten":   return create_baseten_deployment(spec, deployment_key=key)
        case "modal":     return create_modal_deployment(spec, deployment_key=key)
        case "spheron":   return create_spheron_deployment(spec, deployment_key=key)

def _parse_gpu(acceleration: DeploymentAcceleration) -> tuple[DeploymentGPU, int]:
    """
    Parse a deployment acceleration into a GPU type and count.
    """
    count = acceleration[1] if isinstance(acceleration, tuple) else 1
    acceleration = cast(
        Acceleration,
        acceleration[0] if isinstance(acceleration, tuple) else acceleration
    )
    match acceleration:
        case "remote_a100": return "a100", count
        case "remote_h100": return "h100", count
        case "remote_h200": return "h200", count
        case "remote_b200": return "b200", count
        case _: raise ValueError(f"Unsupported deployment acceleration: {acceleration!r}.")

class _SharedDeployment(Deployment):
    """
    Shared deployment served by the Muna inference control plane.
    """

    def wait(self):
        pass

class _DeploymentRecord(BaseModel):
    """
    Deployment record.
    """
    id: str = Field(description="Deployment identifier.")
    key: str | None = Field(None, description="Deployment-specific API key for authentication.")
    endpoint: str | None = Field(None, description="Deployment endpoint.")

class _CreateDeploymentRecordInput(BaseModel):
    tag: str
    name: str
    provider: DeploymentProvider
    kind: DeploymentKind
    gpu: DeploymentGPU
    gpu_count: int = Field(serialization_alias="gpuCount")
    pricing: DeploymentPricing | None

class _UpdateDeploymentRecordInput(BaseModel):
    endpoint: str