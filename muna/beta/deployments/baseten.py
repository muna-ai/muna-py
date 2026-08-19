# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from pydantic import PrivateAttr
from requests import post
from requests.exceptions import RequestException
from tempfile import TemporaryDirectory
from typing import cast, Any

from ._runtime import FXNC_LIBRARY_URL, MUNA_SERVER_URL, SERVER_PORT
from ._schema import Deployment, DeploymentSpec

class BasetenDeployment(Deployment):
    """
    Deployment on Baseten.
    """
    _deployment: Any = PrivateAttr()

    def __init__(self, deployment):
        from truss.api.definitions import ModelDeployment
        from truss.remote.baseten.service import URLConfig
        deployment = cast(ModelDeployment, deployment)
        service = deployment._baseten_service
        dashboard_url = URLConfig.status_page_url(
            service._api.app_url,
            URLConfig.MODEL,
            service.model_id
        )
        super().__init__(
            endpoint_url=service.predict_url,
            dashboard_url=dashboard_url
        )
        self._deployment = deployment

    def wait(self) -> None:
        from truss.api.definitions import ModelDeployment
        deployment = cast(ModelDeployment, self._deployment)
        deployment.wait_for_active()

def create_baseten_deployment(
    spec: DeploymentSpec,
    *,
    deployment_key: str
) -> BasetenDeployment:
    """
    Deploy a predictor to Baseten.
    """
    # Import Truss
    try:
        from truss import push
        from truss.remote.remote_factory import RemoteFactory
    except ImportError as error:
        raise ImportError(
            "The `truss` package is required for Baseten deployments. "
            "Install it with `pip install truss`."
        ) from error
    # Check auth
    if "baseten" not in RemoteFactory.get_available_config_names():
        raise RuntimeError("No Baseten remote is configured. Run `truss login` first.")
    # Create a secret with the deployment key
    secret_name = _baseten_secret_name(deployment_key)
    _upsert_baseten_secret(
        deployment_key,
        secret_name=secret_name,
        remote_factory=RemoteFactory
    )
    # Build Truss config
    config = _build_truss_config(
        spec,
        secret_name=secret_name
    )
    # Deploy
    with TemporaryDirectory() as directory:
        config.write_to_yaml_file(Path(directory) / "config.yaml")
        deployment = push(
            directory,
            remote="baseten",
            publish=True
        )
    # Return
    return BasetenDeployment(deployment)

def _build_truss_config(
    spec: DeploymentSpec,
    *,
    secret_name: str
):
    from truss.base.truss_config import (
        AcceleratorSpec, BaseImage, DockerServer,
        Resources, Runtime, TrussConfig
    )
    start_command = (
        'sh -c "export '
        'LD_LIBRARY_PATH=/app/data '
        'MUNA_HOME=/app/.muna '
        f'MUNA_SERVER_MODELS={spec.tag} '
        f'MUNA_ACCESS_KEY=$(cat /secrets/{secret_name}); '
        'exec /app/data/muna-server"'
    )
    accelerator = (
        AcceleratorSpec(
            accelerator=spec.gpu.upper(),
            count=spec.gpu_count or 1
        )
        if spec.gpu is not None
        else None
    )
    resources = Resources(
        cpu=f"{spec.cpu}" if spec.cpu is not None else "1",
        memory=(
            f"{spec.memory}Mi"
            if spec.memory is not None
            else "2Gi"
        ),
        accelerator=accelerator
    )
    config = TrussConfig(
        model_name=spec.name,
        model_metadata={
            "example_model_input": {
                "model": spec.tag,
                "messages": [
                    {
                        "role": "user",
                        "content": "Say hello in 3 words"
                    }
                ],
                "stream": True
            },
            "tags": ["openai-compatible"]
        },
        base_image=BaseImage(
            image="python:3.13-slim-bookworm"
        ),
        system_packages=["curl"],
        build_commands=[
            "mkdir -p /app/data",
            (
                f"curl -fsSL {MUNA_SERVER_URL} "
                "-o /app/data/muna-server"
            ),
            "chmod +x /app/data/muna-server",
            (
                f"curl -fsSL {FXNC_LIBRARY_URL} "
                "-o /app/data/libFunction.so"
            )
        ],
        docker_server=DockerServer(
            start_command=start_command,
            server_port=SERVER_PORT,
            predict_endpoint="/v1/chat/completions",
            readiness_endpoint="/health",
            liveness_endpoint="/health",
            no_build=False
        ),
        secrets={secret_name: None},
        resources=resources
    )
    if spec.concurrency is not None:
        config.runtime = Runtime(
            predict_concurrency=spec.concurrency
        )
    return config

def _baseten_secret_name(deployment_key: str) -> str:
    digest = sha256(
        deployment_key.encode("utf-8")
    ).hexdigest()[:16].upper()
    return f"MUNA_DEPLOYMENT_{digest}"

def _upsert_baseten_secret(
    deployment_key: str,
    *,
    secret_name: str,
    remote_factory
) -> None:
    """
    Create a secret to hold the deployment key.
    """
    # Retrieve Baseten auth token
    config = remote_factory.load_remote_config("baseten").configs
    provider_key = (
        config.get("api_key") or
        config.get("oauth_access_token")
    )
    if provider_key is None:
        raise RuntimeError(
            "The configured Baseten remote has no usable API credential. "
            "Run `truss login` again."
        )
    # Create secret
    try:
        response = post(
            "https://api.baseten.co/v1/secrets",
            headers={
                "Authorization": f"Bearer {provider_key}"
            },
            json={
                "name": secret_name,
                "value": deployment_key
            },
            timeout=30
        )
        response.raise_for_status()
    except RequestException as error:
        raise RuntimeError(
            "Failed to install the Muna deployment key in Baseten."
        ) from error