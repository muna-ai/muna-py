#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable
from os import environ
from pathlib import Path
from pydantic import BaseModel, Field, TypeAdapter
from requests import request
from requests.exceptions import RequestException
from subprocess import run
from time import sleep, time
from typing import TypeVar
from warnings import warn

from ...logging import CustomProgressTask
from ._runtime import SERVER_PORT
from ._schema import DeploymentSpec
from ._ssh import build_server_script, SSHDeployment

T = TypeVar("T", bound=BaseModel)

class SpheronError(RuntimeError):
    pass

class SpheronDeployment(SSHDeployment):
    """
    Deployment on Spheron GPU cloud.
    """

    def __init__(
        self,
        ip_address: str,
        *,
        ssh_target: list[str]
    ):
        base = f"http://{ip_address}:{SERVER_PORT}"
        super().__init__(
            base,
            ssh_target=ssh_target,
            dashboard_url=_DASHBOARD_URL,
            node_port=SERVER_PORT,
            poll_interval=10,
            node_log_path="~/app/muna-server.log"
        )

def create_spheron_deployment(
    spec: DeploymentSpec,
    *,
    deployment_key: str
) -> SpheronDeployment:
    """
    Deploy a compiled model to a Spheron GPU instance.
    """
    # Check for Spheron API key
    api_key = environ.get("SPHERON_API_KEY")
    if api_key is None:
        raise SpheronError(
            "SPHERON_API_KEY is not set. Create an API key at "
            "https://app.spheron.ai/settings."
        )
    # GPU is required
    if spec.gpu is None:
        raise ValueError("A GPU acceleration is required for Spheron deployments.")
    # Get offers
    with CustomProgressTask(
        "Finding Spheron GPU offers...",
        done_text="Found Spheron GPU offers"
    ):
        offers = _SpheronInstance.get_offers(
            spec.gpu,
            count=spec.gpu_count or 1,
            api_key=api_key
        )
    if not offers:
        raise SpheronError(
            "No available dedicated Spheron offer matches the "
            f"requested {spec.gpu} GPU count ({spec.gpu_count or 1})."
        )
    # Deploy cheapest offer
    offer = min(offers, key=lambda o: o.price)
    with CustomProgressTask(
        "Resolving Spheron team ID...",
        done_text="Resolved Spheron team ID"
    ):
        team_id = _resolve_spheron_team_id(api_key=api_key)
    private_key_path, public_key = _discover_ssh_key()
    with CustomProgressTask(
        "Creating Spheron instance...",
        done_text="Created Spheron instance"
    ):
        deployment = _SpheronInstance.create(
            offer,
            public_key=public_key,
            name=spec.name,
            team_id=team_id,
            api_key=api_key
        )
    try:
        # Wait until we have an IP address
        with CustomProgressTask(
            "Waiting for Spheron instance...",
            done_text="Spheron instance is running"
        ):
            _wait_until(
                lambda: deployment.ip_address(),
                poll_interval=_POLL_INTERVAL,
                timeout=_INSTANCE_TIMEOUT
            )
        # Connect via SSH
        ip_address = deployment.ip_address()
        ssh_target = [
            "-i", str(private_key_path),
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            f"{deployment.ssh_user()}@{ip_address}"
        ]
        with CustomProgressTask(
            "Connecting to Spheron instance...",
            done_text="Connected to Spheron instance"
        ):
            _wait_until(
                lambda: run(
                    ["ssh", *ssh_target, "true"],
                    capture_output=True,
                    text=True
                ).returncode == 0,
                poll_interval=_POLL_INTERVAL,
                timeout=_SSH_TIMEOUT
            )
        # Setup muna-server
        with CustomProgressTask(
            "Installing muna-server...",
            done_text="Installed muna-server"
        ):
            result = run(
                ["ssh", *ssh_target, "bash -s"],
                input=build_server_script(
                    tag=spec.tag,
                    access_key=deployment_key,
                    install_dir="$HOME/app"
                ),
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise SpheronError(
                    "Failed to install and launch muna-server on the "
                    f"Spheron instance (SSH exited with code {result.returncode})."
                    f" {result.stderr.strip()}"
                )
    except BaseException:
        try:
            deployment.terminate()
        except SpheronError as error:
            warn(
                "Could not terminate the partially configured Spheron "
                f"deployment {deployment.id}: {error}",
                stacklevel=2
            )
        raise
    # Return
    return SpheronDeployment(
        ip_address,
        ssh_target=ssh_target
    )

def _request(
    method: str,
    path: str,
    *,
    api_key: str,
    body: BaseModel | None = None,
    params: dict[str, object] | None = None,
    response_type: type[T] | None = None,
    timeout: float = 30
) -> T:
    try:
        response = request(
            method,
            f"{_API_URL}{path}",
            headers={ "Authorization": f"Bearer {api_key}" },
            json=(
                body.model_dump(mode="json", by_alias=True)
                if body is not None
                else None
            ),
            params=params,
            timeout=timeout
        )
    except RequestException as error:
        raise SpheronError(
            f"Spheron API request failed: {error}"
        ) from error
    if not response.ok:
        raise SpheronError(
            f"Spheron API {method} {path} returned status "
            f"{response.status_code}: {response.text[:200]}"
        )
    data = response.json() if response.content else {}
    return (
        TypeAdapter(response_type).validate_python(data)
        if response_type
        else data
    )

def _resolve_spheron_team_id(*, api_key: str) -> str:
    response = _request(
        "GET",
        "/balance",
        api_key=api_key,
        response_type=_SpheronBalanceResponse
    )
    current = next((
        team
        for team in response.teams
        if team.is_current_team
    ), None)
    if current is None:
        current = next(iter(response.teams), None)
    team_id = (
        current.team_id
        if current is not None
        else None
    )
    if not isinstance(team_id, str):
        raise SpheronError("Could not resolve a Spheron team ID.")
    return team_id

def _wait_until(
    cond: Callable[[], object],
    *,
    poll_interval: float,
    timeout: float
) -> None:
    deadline = time() + timeout
    while time() < deadline:
        if cond():
            return
        sleep(poll_interval)
    raise SpheronError("Timed out waiting for condition")

def _discover_ssh_key() -> tuple[Path, str]:
    for private_key_path in (
        Path.home() / ".ssh" / "id_ed25519",
        Path.home() / ".ssh" / "id_rsa"
    ):
        public_key_path = private_key_path.with_name(
            f"{private_key_path.name}.pub"
        )
        if (
            private_key_path.is_file()
            and public_key_path.is_file()
        ):
            return (
                private_key_path,
                public_key_path.read_text().strip()
            )
    raise SpheronError("No SSH keypair found at ~/.ssh/id_ed25519 or ~/.ssh/id_rsa.")

class _SpheronInstance:

    def __init__(self, id: str, api_key: str) -> None:
        self.id = id
        self.__api_key = api_key

    def info(self) -> _SpheronDeploymentResponse:
        return _request(
            "GET",
            f"/deployments/{self.id}",
            api_key=self.__api_key,
            response_type=_SpheronDeploymentResponse
        )

    def ip_address(self) -> str | None:
        instance = self.info()
        status = instance.status.lower()
        if status in { "failed", "terminated", "terminated-provider" }:
            raise SpheronError(
                f"Instance entered status {status!r} while provisioning."
            )
        return instance.ip_address

    def ssh_user(self) -> str:
        instance = self.info()
        if instance.user:
            return instance.user
        if instance.ssh_command:
            for token in instance.ssh_command.split():
                if "@" in token:
                    return token.split("@", maxsplit=1)[0]
        return "ubuntu"

    def terminate(self):
        _request(
            "DELETE",
            f"/deployments/{self.id}",
            api_key=self.__api_key,
        )

    @classmethod
    def create(
        cls,
        offer: _SpheronGPUOffer,
        *,
        public_key: str,
        name: str,
        team_id: str,
        api_key: str
    ) -> _SpheronInstance:
        response = _request(
            "POST",
            "/deployments",
            api_key=api_key,
            body=_CreateSpheronDeploymentInput(
                provider=offer.provider,
                offer_id=offer.offer_id,
                gpu_type=offer.gpu_type,
                gpu_count=offer.gpu_count,
                region=offer.clusters[0],
                operating_system=offer.os_options[0],
                instance_type=offer.instance_type,
                ssh_public_key=public_key,
                name=name,
                team_id=team_id
            ),
            response_type=_SpheronDeploymentResponse,
            timeout=_INSTANCE_TIMEOUT
        )
        return cls(response.id, api_key)

    @classmethod
    def get_offers(
        cls,
        gpu: str,
        count: int,
        api_key: str
    ) -> list[_SpheronGPUOffer]:
        response = _request(
            "GET",
            "/gpu-offers",
            api_key=api_key,
            params={
                "search": gpu,
                "limit": 50,
                #"instanceType": "DEDICATED"
            },
            response_type=_SpheronGPUOffersResponse
        )
        return [
            offer.model_copy(update={ "gpu_type": group.gpu_type })
            for group in response.data if group.gpu_type.lower().startswith(gpu)
            for offer in group.offers if (
                offer.available and
                offer.gpu_count == count and
                offer.clusters and
                offer.os_options
            )
        ]

class _SpheronDeploymentResponse(BaseModel):
    id: str
    status: str = ""
    ip_address: str | None = Field(None, validation_alias="ipAddress")
    user: str | None = None
    ssh_command: str | None = Field(None, validation_alias="sshCommand")

class _SpheronGPUOffer(BaseModel):
    gpu_type: str = Field("")
    offer_id: str = Field(validation_alias="offerId")
    provider: str
    gpu_count: int = Field(validation_alias="gpuCount")
    price: float
    available: bool
    instance_type: str = Field(validation_alias="instanceType")
    clusters: list[str]
    os_options: list[str]

class _SpheronGPUOfferGroup(BaseModel):
    gpu_type: str = Field(validation_alias="gpuType")
    offers: list[_SpheronGPUOffer] = Field(default_factory=list)

class _SpheronGPUOffersResponse(BaseModel):
    data: list[_SpheronGPUOfferGroup] = Field(default_factory=list)

class _SpheronTeam(BaseModel):
    team_id: str | None = Field(None, validation_alias="teamId")
    is_current_team: bool = Field(False, validation_alias="isCurrentTeam")

class _SpheronBalanceResponse(BaseModel):
    teams: list[_SpheronTeam] = Field(default_factory=list)

class _CreateSpheronDeploymentInput(BaseModel):
    provider: str
    offer_id: str = Field(serialization_alias="offerId")
    gpu_type: str = Field(serialization_alias="gpuType")
    gpu_count: int = Field(serialization_alias="gpuCount")
    region: str
    operating_system: str = Field(serialization_alias="operatingSystem")
    instance_type: str = Field(serialization_alias="instanceType")
    ssh_public_key: str
    name: str
    team_id: str = Field(serialization_alias="teamId")

_API_URL = "https://app.spheron.ai/api"
_DASHBOARD_URL = "https://app.spheron.ai"
_INSTANCE_TIMEOUT = 10 * 60
_SSH_TIMEOUT = 5 * 60
_POLL_INTERVAL = 10