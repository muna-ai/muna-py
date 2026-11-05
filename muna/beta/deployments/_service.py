# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel, Field
from urllib.parse import quote

from ...api import MunaAPIError, MunaClient
from ._schema import Deployment, DeploymentAccess, DeploymentPricing

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
        access: DeploymentAccess = "private",
        pricing: DeploymentPricing | None = None
    ) -> Deployment:
        """
        Deploy a predictor, or update the existing deployment of the predictor.

        Parameters:
            tag (str): Predictor tag.
            access (DeploymentAccess): Deployment access mode. Only superusers can create non-private deployments.
            pricing (DeploymentPricing): Prices that callers pay. Only superusers can set pricing.

        Returns:
            Deployment: Deployment.
        """
        body = _CreateDeploymentInput(
            tag=tag,
            access=access,
            pricing=pricing
        )
        return self.client.request(
            method="POST",
            path="/deployments",
            body=body.model_dump(mode="json", by_alias=True, exclude_none=True),
            response_type=Deployment
        )

    def retrieve(self, tag: str) -> Deployment | None:
        """
        Retrieve the deployment of a predictor.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            Deployment: Deployment, or `None` if the predictor is not deployed.
        """
        page = self.client.request(
            method="GET",
            path=f"/deployments?tag={quote(tag, safe='')}&limit=1",
            response_type=_DeploymentPage
        )
        return page.data[0] if page.data else None

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
        deployment = self.retrieve(tag)
        if deployment is None:
            return False
        try:
            self.client.request(
                method="DELETE",
                path=f"/deployments/{deployment.id}"
            )
        except MunaAPIError as error:
            if error.status_code == 404:
                return False
            raise
        return True

_PAGE_SIZE = 100

class _CreateDeploymentInput(BaseModel):
    tag: str
    kind: str = "shared"
    access: DeploymentAccess
    pricing: DeploymentPricing | None

class _Pagination(BaseModel):
    total: int = Field(description="Total number of results.")

class _DeploymentPage(BaseModel):
    data: list[Deployment]
    pagination: _Pagination
