# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable
from datetime import timedelta
from math import ceil
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing import Annotated, Literal
from urllib.parse import quote

from ...api import MunaAPIError, MunaClient
from .schema import (
    Deployment, DeploymentPricing, DeploymentProgress,
    DeploymentTenancy, parse_max_queue_time
)

class DeploymentError(RuntimeError):
    """
    Preparation failed or its stream ended without confirmed readiness.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        deployment: Deployment | None = None
    ):
        super().__init__(message)
        self.code = code
        self.deployment = deployment

class DeploymentService:
    """
    Deploy compiled models on Muna inference endpoints.
    """

    def __init__(self, client: MunaClient):
        self.client = client

    def create(
        self,
        tag: str,
        *,
        tenancy: DeploymentTenancy | None = None,
        max_queue_time: timedelta | str | None = None,
        listed: bool | None = None,
        pricing: DeploymentPricing | None = None,
        progress: Callable[[DeploymentProgress], None] | None = None
    ) -> Deployment:
        """
        Deploy a predictor and wait for a complete disk copy on an eligible node.

        Parameters:
            tag (str): Predictor tag.
            tenancy (DeploymentTenancy): Whether the deployment shares GPUs with other deployments, or gets full GPU performance. `None` preserves the existing setting, or uses the platform default on creation.
            max_queue_time (timedelta | str): Maximum time a request waits for compute before it is rejected, like '30s' or '10m' (at most 10 minutes). `None` preserves the existing setting, or uses the platform default on creation.
            listed (bool): Whether a public predictor is listed in the public catalog. Only superusers can set listing. `None` preserves the existing setting.
            pricing (DeploymentPricing): Prices that callers pay. Only superusers can set pricing.
            progress (Callable): Optional callback for reduced preparation progress.

        Returns:
            Deployment: Deployment with confirmed initial disk readiness.

        Raises:
            DeploymentError: Preparation failed, or the stream ended without readiness.
            MunaAPIError: The API rejected the request before streaming.
        """
        body = _CreateDeploymentInput(
            tag=tag,
            listed=listed,
            tenancy=tenancy,
            max_queue_time=(
                parse_max_queue_time(max_queue_time)
                if max_queue_time is not None
                else None
            ),
            pricing=pricing
        )
        payload = body.model_dump(mode="json", by_alias=True, exclude_none=True)
        deployment: Deployment | None = None
        while True:
            events = self.client.stream(
                method="POST",
                path="/deployments",
                body=payload,
                response_type=DeploymentEvent
            )
            pending = False
            try:
                for event in events:
                    if event.event in ("deployment", "ready"):
                        snapshot = event.data
                        if snapshot.tag.lower() != tag.lower() or (deployment and snapshot.id != deployment.id):
                            raise DeploymentError(
                                "Deployment stream changed identity.",
                                code="invalid_stream",
                                deployment=deployment
                            )
                        deployment = snapshot
                        if event.event == "ready":
                            if not deployment.status or deployment.status.preparation.state != "ready":
                                raise DeploymentError(
                                    "Deployment stream did not confirm disk readiness.",
                                    code="invalid_stream",
                                    deployment=deployment
                                )
                            return deployment
                        if progress and deployment.status and deployment.status.progress:
                            progress(deployment.status.progress)
                    elif event.event == "progress":
                        if progress:
                            progress(event.data)
                    elif event.event == "error":
                        raise DeploymentError(
                            event.data.error,
                            code=event.data.code,
                            deployment=deployment
                        )
                    elif event.event == "pending":
                        if not deployment or event.data.id != deployment.id or event.data.tag.lower() != tag.lower():
                            raise DeploymentError(
                                "Deployment resume identity did not match.",
                                code="invalid_stream",
                                deployment=deployment
                            )
                        pending = True
                        break
            finally:
                # Close HTTP even on ready, callback exceptions, or Ctrl-C.
                if close := getattr(events, "close", None):
                    close()
            if not pending:
                raise DeploymentError(
                    "Deployment progress ended before readiness was confirmed. Repeat the deployment request to resume watching.",
                    code="stream_interrupted", deployment=deployment
                )

    def retrieve(self, tag: str) -> Deployment | None:
        """
        Retrieve the deployment of a predictor.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            Deployment: Deployment, or `None` if the predictor is not deployed.
        """
        try:
            return self.client.request(
                method="GET",
                path=f"/deployments/{quote(tag, safe='@/')}",
                response_type=Deployment
            )
        except MunaAPIError as error:
            if error.status_code == 404:
                return None
            raise

    def list(self) -> list[Deployment]:
        """
        List your deployments.

        Returns:
            list: Deployments.
        """
        deployments: list[Deployment] = []
        while True:
            page = self.client.request(
                method="GET",
                path=f"/deployments?offset={len(deployments)}&limit={_PAGE_SIZE}",
                response_type=_DeploymentPage
            )
            deployments.extend(page.data)
            if not page.data or len(deployments) >= page.pagination.total:
                return deployments

    def delete(self, tag: str) -> bool:
        """
        Delete the deployment of a predictor.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            bool: Whether the deployment was deleted.
        """
        try:
            self.client.request(
                method="DELETE",
                path=f"/deployments/{quote(tag, safe='@/')}"
            )
        except MunaAPIError as error:
            if error.status_code == 404:
                return False
            raise
        return True

_PAGE_SIZE = 100

class _CreateDeploymentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str
    tenancy: DeploymentTenancy | None = None
    max_queue_time: timedelta | None = Field(None, serialization_alias="maxQueueTime")
    listed: bool | None = None
    pricing: DeploymentPricing | None = None

    @field_validator("max_queue_time")
    @classmethod
    def _validate_max_queue_time(cls, value: timedelta | None) -> timedelta | None:
        return parse_max_queue_time(value) if value is not None else None

    @field_serializer("max_queue_time")
    def _serialize_max_queue_time(self, value: timedelta | None) -> int | None:
        return ceil(value.total_seconds()) if value is not None else None

class _Pagination(BaseModel):
    total: int = Field(description="Total number of results.")

class _DeploymentPage(BaseModel):
    data: list[Deployment]
    pagination: _Pagination

class DeploymentSnapshotEvent(BaseModel):
    event: Literal["deployment", "ready"]
    data: Deployment

class DeploymentProgressEvent(BaseModel):
    event: Literal["progress"]
    data: DeploymentProgress

class _ErrorData(BaseModel):
    code: str
    error: str

class DeploymentErrorEvent(BaseModel):
    event: Literal["error"]
    data: _ErrorData

class _PendingData(BaseModel):
    id: str
    tag: str
    message: str

class DeploymentPendingEvent(BaseModel):
    event: Literal["pending"]
    data: _PendingData

DeploymentEvent = Annotated[
    DeploymentSnapshotEvent |
    DeploymentProgressEvent |
    DeploymentErrorEvent    |
    DeploymentPendingEvent,
    Field(discriminator="event")
]
