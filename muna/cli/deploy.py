# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from pydantic import BaseModel, Field
from requests import get, post
from requests.exceptions import RequestException
from rich import print
from shlex import quote, split
from subprocess import Popen, run
from sys import version_info
from tempfile import TemporaryDirectory
from time import sleep, time
from typer import Argument, Exit, Option
from typing import Annotated, Literal, Protocol

from ..muna import Muna
from .auth import get_access_key

DeploymentProvider = Literal["baseten", "modal", "baremetal"]
DeploymentGPU = Literal["a100", "h100", "h200", "b200"]
DeploymentPricingKind = Literal["tokens", "images", "duration"]

def deploy_function(
    tag: Annotated[str, Argument(help="Predictor tag.")],
    provider: Annotated[
        DeploymentProvider,
        Option(help="Cloud to deploy the predictor to.")
    ],
    name: Annotated[
        str | None,
        Option(help="Deployed model name.")
    ] = None,
    cpu: Annotated[int | None, Option(
        help="Number of vCPUs to request.",
        min=0,
    )] = None,
    gpu: Annotated[
        DeploymentGPU | None,
        Option(help="GPU hardware configuration.")
    ] = None,
    gpu_count: Annotated[
        int | None,
        Option(help="Number of GPUs to request.")
    ] = None,
    memory: Annotated[int | None, Option(
        help="Memory to request in MB.",
        min=16
    )] = None,
    concurrency: Annotated[int | None, Option(
        help="Maximum concurrent requests sent to an instance.",
        min=1
    )] = None,
    min_replicas: Annotated[int | None, Option(
        help="Minimum replicas for autoscaling.",
        min=0
    )] = None,
    max_replicas: Annotated[int | None, Option(
        help="Maximum replicas for autoscaling.",
        min=1
    )] = None,
    scaledown_window: Annotated[float | None, Option(
        help="Autoscaling scale down window in seconds.",
        min=0
    )] = None,
    ssh_host: Annotated[str | None, Option(
        help="SSH target for --provider baremetal, e.g. 'root@1.2.3.4 -p 22 -i ~/.ssh/key'. Accepts the full SSH target string (everything after `ssh`).")
    ] = None,
    endpoint_url: Annotated[str | None, Option(
        help="Public HTTP(S) base URL where the deployed server is reachable, e.g. 'https://<pod-id>-8000.proxy.runpod.net'. Required for --provider baremetal.")
    ] = None,
    shared: Annotated[bool, Option(
        "--shared",
        hidden=True,
        help="Create a shared endpoint. Requires superuser access.")
    ] = False,
    pricing_kind: Annotated[DeploymentPricingKind | None, Option(
        "--pricing",
        hidden=True,
        help="Shared endpoint pricing kind.")
    ] = None,
    input_price: Annotated[float | None, Option(
        "--input-price",
        hidden=True,
        min=0,
        help="Price per million input tokens, image, or minute in USD.")
    ] = None,
    output_price: Annotated[float | None, Option(
        "--output-price",
        hidden=True,
        min=0,
        help="Output price per million tokens in USD.")
    ] = None,
    dry_run: Annotated[bool, Option(
        "--dry-run",
        help="Generate the generated deployment artifact instead of creating a deployment."
    )] = False,
    wait: Annotated[bool, Option(
        "--wait",
        help="Whether to wait until the deployment is complete."
    )] = False
):
    # Ensure that the user has access to the predictor
    muna = Muna(get_access_key())
    predictor = muna.predictors.retrieve(tag)
    if predictor is None:
        print(
            f"[bold red]Error:[/bold red] Predictor [bold cyan]{tag}[/bold cyan] was not found "
            "or you do not have access to it. Make sure you are signed in to the Muna "
            "CLI with [bold orange1]muna auth login <access key>[/bold orange1]."
        )
        raise Exit(code=1)
    # Create deployment
    pricing = _build_deployment_pricing(
        shared=shared,
        kind=pricing_kind,
        input_price=input_price,
        output_price=output_price
    )
    spec = _DeploymentSpec(
        tag=tag,
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
        shared=shared,
        pricing=pricing
    )
    deployment = _create_deployment(
        spec,
        provider=provider,
        dry_run=dry_run,
        muna=muna
    )
    # Log
    if deployment.dashboard_url:
        print(f"Track deployment progress at [link={deployment.dashboard_url}][bold cyan]{deployment.dashboard_url}[/bold cyan][/link]")
    if wait:
        deployment.wait()
    if deployment.endpoint_url:
        print(f"Endpoint available at [link={deployment.endpoint_url}][bold cyan]{deployment.endpoint_url}[/bold cyan][/link]")

def _create_deployment(
    spec: _DeploymentSpec,
    *,
    provider: DeploymentProvider,
    dry_run: bool,
    muna: Muna
) -> _Deployment:
    match provider:
        case "baseten":     return _create_deployment_baseten(spec, dry_run=dry_run, muna=muna)
        case "modal":       return _create_deployment_modal(spec, dry_run=dry_run, muna=muna)
        case "baremetal":   return _create_deployment_baremetal(spec, dry_run=dry_run, muna=muna)

def _create_deployment_baseten(
    spec: _DeploymentSpec,
    *,
    dry_run: bool,
    muna: Muna
) -> _Deployment:
    try:
        from truss import push
        from truss.remote.remote_factory import RemoteFactory
    except ImportError:
        print(
            "[bold red]Error:[/bold red] The `truss` package is required to deploy to Baseten. "
            "Install it with [bold]pip install truss[/bold]."
        )
        raise Exit(code=1)
    if dry_run:
        truss_config = _build_truss_config(
            spec,
            secret_name="MUNA_DEPLOYMENT_KEY"
        )
        predictor_slug = spec.tag.lstrip("@").replace("/", "_")
        output_directory = Path(f"{predictor_slug}-truss")
        output_directory.mkdir(parents=True, exist_ok=True)
        config_path = output_directory / "config.yaml"
        truss_config.write_to_yaml_file(config_path)
        print(f"Wrote Truss config to [bold cyan]{config_path}[/bold cyan]")
        return _DryRunDeployment()
    if "baseten" not in RemoteFactory.get_available_config_names():
        print(
            "[bold red]Error:[/bold red] No Baseten remote is configured. "
            "Run [bold]truss login[/bold] to authenticate with Baseten."
        )
        raise Exit(code=1)
    deployment_record = _create_deployment_record(
        muna,
        spec=spec,
        provider="baseten"
    )
    deployment_key = deployment_record.key
    secret_name = _baseten_secret_name(deployment_key)
    truss_config = _build_truss_config(spec, secret_name=secret_name)
    _upsert_baseten_secret(
        deployment_key,
        secret_name=secret_name,
        remote_factory=RemoteFactory
    )
    with TemporaryDirectory() as temp_directory:
        truss_config.write_to_yaml_file(Path(temp_directory) / "config.yaml")
        deployment = push(temp_directory, remote="baseten", publish=True)
    result = _BasetenDeployment(deployment)
    _update_deployment_endpoint(
        muna,
        deployment_id=deployment_record.id,
        endpoint=result.endpoint_url
    )
    return result

def _create_deployment_modal(
    spec: _DeploymentSpec,
    *,
    dry_run: bool,
    muna: Muna
) -> _Deployment:
    if dry_run:
        print(
            "[bold red]Error:[/bold red] Dry runs are not supported for Modal deployments. "
            "Modal defines deployments in Python code rather than a declarative artifact, "
            "so there is nothing to generate without deploying."
        )
        raise Exit(code=1)
    try:
        from modal import enable_output, web_server, App, Image, Secret, Volume
    except ImportError:
        print(
            "[bold red]Error:[/bold red] The `modal` package is required to deploy to Modal. "
            "Install it with [bold]pip install modal[/bold]."
        )
        raise Exit(code=1)
    deployment_record = _create_deployment_record(
        muna,
        spec=spec,
        provider="modal"
    )
    deployment_key = deployment_record.key
    predictor_slug = spec.tag.lstrip("@").replace("/", "_")
    app = App(f"muna-{predictor_slug}")
    volume = Volume.from_name("muna-deploy-cache", create_if_missing=True, version=2)
    image = (Image
        .debian_slim(python_version=f"{version_info.major}.{version_info.minor}")
        .apt_install("curl")
        .run_commands( # download muna-server from GitHub and libFunction
            f"mkdir -p /app && "
            f"curl -fsSL {_MUNA_SERVER_URL} -o /app/muna-server && "
            f"chmod +x /app/muna-server && "
            f"curl -fsSL {_FXNC_LIBRARY_URL} -o /app/libFunction.so"
        )
    )
    @app.function(
        image=image,
        cpu=spec.cpu,
        gpu=f"{spec.gpu}:{spec.gpu_count or 1}" if spec.gpu is not None else None,
        memory=spec.memory,
        min_containers=spec.min_replicas,
        max_containers=spec.max_replicas,
        volumes={ "/muna": volume },
        env={
            "LD_LIBRARY_PATH": "/app",
            "MUNA_HOME": "/muna"
        },
        secrets=[
            Secret.from_dict({ "MUNA_ACCESS_KEY": deployment_key })
        ],
        timeout=60 * 60,
        startup_timeout=45 * 60,
        scaledown_window=spec.scaledown_window,
        serialized=True
    )
    @web_server(8000, startup_timeout=45 * 60)
    def serve():
        Popen(["/app/muna-server"])
    with enable_output():
        app.deploy()
    result = _ModalDeployment(app, serve)
    _update_deployment_endpoint(
        muna,
        deployment_id=deployment_record.id,
        endpoint=result.endpoint_url
    )
    return result

def _create_deployment_baremetal(
    spec: _DeploymentSpec,
    *,
    dry_run: bool,
    muna: Muna
) -> _Deployment:
    if not spec.ssh_host:
        print(
            "[bold red]Error:[/bold red] [bold]--ssh-host[/bold] is required for "
            "[bold]--provider baremetal[/bold]. Pass the full SSH target, e.g. "
            "[bold]--ssh-host \"root@1.2.3.4 -p 22 -i ~/.ssh/key\"[/bold]."
        )
        raise Exit(code=1)
    if not spec.endpoint_url:
        print(
            "[bold red]Error:[/bold red] [bold]--endpoint-url[/bold] is required for "
            "[bold]--provider baremetal[/bold]. Pass the public URL where the server "
            "will be reachable, e.g. [bold]--endpoint-url https://<pod-id>-8000.proxy.runpod.net[/bold]."
        )
        raise Exit(code=1)
    # Warn that resource / autoscaling flags are meaningless for a fixed node
    ignored = [
        name
        for name, value in {
            "--cpu": spec.cpu,
            "--gpu": spec.gpu,
            "--gpu-count": spec.gpu_count,
            "--memory": spec.memory,
            "--concurrency": spec.concurrency,
            "--min-replicas": spec.min_replicas,
            "--max-replicas": spec.max_replicas,
            "--scaledown-window": spec.scaledown_window,
        }.items()
        if value is not None
    ]
    if ignored:
        print(
            f"[bold yellow]Warning:[/bold yellow] Ignoring resource/autoscaling flags "
            f"([bold]{', '.join(ignored)}[/bold]) which do not apply to a fixed baremetal node."
        )
    if dry_run:
        script = _build_baremetal_script(
            tag=spec.tag,
            access_key="MUNA_DEPLOYMENT_KEY"
        )
        print(script)
        return _DryRunDeployment()
    deployment_record = _create_deployment_record(
        muna,
        spec=spec,
        provider="baremetal"
    )
    deployment_key = deployment_record.key
    # Build the remote setup + launch script
    script = _build_baremetal_script(
        tag=spec.tag,
        access_key=deployment_key
    )
    # Install and launch muna-server over SSH (blocks through the preload/download)
    ssh_target = split(spec.ssh_host)
    print(f"Installing [bold cyan]muna-server[/bold cyan] and preloading [bold cyan]{spec.tag}[/bold cyan] on the node...")
    result = run(["ssh", *ssh_target, "bash -s"], input=script, text=True)
    if result.returncode != 0:
        print(
            "[bold red]Error:[/bold red] Failed to install and launch muna-server on the node "
            f"(ssh exited with code [bold]{result.returncode}[/bold]). Check the SSH connection and node logs."
        )
        raise Exit(code=1)
    # Create baremetal deployment
    deployment = _BaremetalDeployment(spec.endpoint_url, ssh_target=ssh_target)
    _update_deployment_endpoint(
        muna,
        deployment_id=deployment_record.id,
        endpoint=deployment.endpoint_url
    )
    return deployment

def _build_deployment_pricing(
    *,
    shared: bool,
    kind: DeploymentPricingKind | None,
    input_price: float | None,
    output_price: float | None
) -> _DeploymentPricing | None:
    has_pricing_option = any(
        value is not None
        for value in (kind, input_price, output_price)
    )
    if not shared:
        if has_pricing_option:
            print("[bold red]Error:[/bold red] Pricing options require [bold]--shared[/bold].")
            raise Exit(code=1)
        return None
    match kind:
        case None:
            print("[bold red]Error:[/bold red] Shared endpoints require [bold]--pricing[/bold].")
            raise Exit(code=1)
        case "tokens":
            if input_price is None:
                print("[bold red]Error:[/bold red] Token pricing requires [bold]--input-price[/bold].")
                raise Exit(code=1)
            return _TokenDeploymentPricing(
                input_per_million=input_price,
                output_per_million=output_price
            )
        case "images":
            if output_price is not None:
                print("[bold red]Error:[/bold red] Image pricing does not accept [bold]--output-price[/bold].")
                raise Exit(code=1)
            if input_price is None:
                print("[bold red]Error:[/bold red] Image pricing requires [bold]--input-price[/bold].")
                raise Exit(code=1)
            return _ImageDeploymentPricing(per_image=input_price)
        case "duration":
            if output_price is not None:
                print("[bold red]Error:[/bold red] Duration pricing does not accept [bold]--output-price[/bold].")
                raise Exit(code=1)
            if input_price is None:
                print("[bold red]Error:[/bold red] Duration pricing requires [bold]--input-price[/bold].")
                raise Exit(code=1)
            return _DurationDeploymentPricing(per_minute=input_price)

def _build_baremetal_script(*, tag: str, access_key: str) -> str:
    access_key = quote(access_key)
    tag = quote(tag)
    serve_command = (
        'echo $$ > "$DIR/muna-server.pid"; '
        f'exec env LD_LIBRARY_PATH="$DIR" MUNA_HOME="$DIR/.muna" '
        f"MUNA_ACCESS_KEY={access_key} PORT={_BAREMETAL_PORT} "
        '"$DIR/muna-server" serve'
    )
    return (
        f"set -e\n"
        f"export DIR=/app\n"
        f"mkdir -p \"$DIR\"\n"
        f"curl -fsSL {_MUNA_SERVER_URL} -o \"$DIR/muna-server\" && chmod +x \"$DIR/muna-server\"\n"
        f"curl -fsSL {_FXNC_LIBRARY_URL} -o \"$DIR/libFunction.so\"\n"
        f"# preload weights up front (download only) so the first request skips the download;\n"
        f"# uses the same MUNA_HOME as serve so the cache is shared.\n"
        f"env LD_LIBRARY_PATH=\"$DIR\" MUNA_HOME=\"$DIR/.muna\" MUNA_ACCESS_KEY={access_key} "
        f"\"$DIR/muna-server\" preload {tag}\n"
        f"# stop a previous instance if present\n"
        f"[ -f \"$DIR/muna-server.pid\" ] && kill \"$(cat \"$DIR/muna-server.pid\")\" 2>/dev/null || true\n"
        f"# detach; `exec` keeps muna-server on the same PID we record\n"
        f"setsid bash -c {quote(serve_command)} "
        f"> \"$DIR/muna-server.log\" 2>&1 </dev/null &\n"
    )

def _build_truss_config(spec: _DeploymentSpec, *, secret_name: str):
    from truss.base.truss_config import (
        AcceleratorSpec, BaseImage, DockerServer,
        Resources, Runtime, TrussConfig
    )
    START_COMMAND = (
        'sh -c "export PORT=8000 LD_LIBRARY_PATH=/app/data MUNA_HOME=/app/.fxn '
        f'MUNA_ACCESS_KEY=$(cat /secrets/{secret_name}); '
        'exec /app/data/muna-server"'
    )
    accelerator = (
        AcceleratorSpec(
            accelerator=spec.gpu.upper(),
            count=spec.gpu_count
        )
        if spec.gpu is not None
        else None
    )
    resources = Resources(
        cpu=f"{spec.cpu}" if spec.cpu is not None else "1",
        memory=f"{spec.memory}Mi" if spec.memory is not None else "2Gi",
        accelerator=accelerator
    )
    config = TrussConfig(
        model_name=spec.name,
        model_metadata={
            "example_model_input": {
                "model": spec.tag,
                "messages": [{ "role": "user", "content": "Say hello in 3 words" }],
                "stream": True,
            },
            "tags": ["openai-compatible"],
        },
        base_image=BaseImage(image="python:3.13-slim-bookworm"),
        system_packages=["curl"],
        build_commands=[
            "mkdir -p /app/data",
            f"curl -fsSL {_MUNA_SERVER_URL} -o /app/data/muna-server",
            "chmod +x /app/data/muna-server",
            f"curl -fsSL {_FXNC_LIBRARY_URL} -o /app/data/libFunction.so",
        ],
        docker_server=DockerServer(
            start_command=START_COMMAND,
            server_port=8000,
            predict_endpoint="/v1/chat/completions",
            readiness_endpoint="/health",
            liveness_endpoint="/health",
            no_build=False # not yet
        ),
        #environment_variables={ "MUNA_PREDICTOR_TAG": spec.tag }, # this is technically useless
        secrets={ secret_name: None },
        resources=resources,
    )
    if spec.concurrency is not None:
        config.runtime = Runtime(predict_concurrency=spec.concurrency)
    return config

def _create_deployment_record(
    muna: Muna,
    *,
    spec: _DeploymentSpec,
    provider: DeploymentProvider
) -> _DeploymentRecord:
    gpu_count = (
        spec.gpu_count
        if spec.gpu_count is not None
        else (1 if spec.gpu is not None else None)
    )
    body: dict[str, object] = {
        "tag": spec.tag,
        "name": spec.name,
        "provider": provider,
        "kind": "shared" if spec.shared else "dedicated",
        "gpu": spec.gpu,
        "gpuCount": gpu_count,
    }
    if spec.pricing is not None:
        body["pricing"] = spec.pricing.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=True
        )
    return muna.client.request(
        method="POST",
        path="/deployments",
        body=body,
        response_type=_DeploymentRecord
    )

def _update_deployment_endpoint(
    muna: Muna,
    *,
    deployment_id: str,
    endpoint: str | None
) -> None:
    if endpoint is None:
        return
    muna.client.request(
        method="PATCH",
        path=f"/deployments/{deployment_id}",
        body={ "endpoint": endpoint },
        response_type=_DeploymentEndpointRecord
    )

def _baseten_secret_name(deployment_key: str) -> str:
    digest = sha256(deployment_key.encode("utf-8")).hexdigest()[:16].upper()
    return f"MUNA_DEPLOYMENT_{digest}"

def _upsert_baseten_secret(
    deployment_key: str,
    *,
    secret_name: str,
    remote_factory
) -> None:
    config = remote_factory.load_remote_config("baseten").configs
    provider_key = config.get("api_key") or config.get("oauth_access_token")
    if not provider_key:
        print(
            "[bold red]Error:[/bold red] The configured Baseten remote has no "
            "usable API credential. Run [bold]truss login[/bold] again."
        )
        raise Exit(code=1)
    try:
        response = post(
            "https://api.baseten.co/v1/secrets",
            headers={ "Authorization": f"Bearer {provider_key}" },
            json={ "name": secret_name, "value": deployment_key },
            timeout=30
        )
        response.raise_for_status()
    except RequestException as error:
        print(
            "[bold red]Error:[/bold red] Failed to install the Muna deployment "
            f"key in Baseten: {error}"
        )
        raise Exit(code=1)

class _TokenDeploymentPricing(BaseModel):
    kind: Literal["tokens"] = "tokens"
    currency: Literal["USD"] = "USD"
    input_per_million: float = Field(ge=0, serialization_alias="inputPerMillion")
    output_per_million: float | None = Field(None, ge=0, serialization_alias="outputPerMillion")

class _ImageDeploymentPricing(BaseModel):
    kind: Literal["images"] = "images"
    currency: Literal["USD"] = "USD"
    per_image: float = Field(ge=0, serialization_alias="perImage")

class _DurationDeploymentPricing(BaseModel):
    kind: Literal["duration"] = "duration"
    currency: Literal["USD"] = "USD"
    per_minute: float = Field(ge=0, serialization_alias="perMinute")

_DeploymentPricing = (
    _TokenDeploymentPricing |
    _ImageDeploymentPricing |
    _DurationDeploymentPricing
)

class _DeploymentRecord(BaseModel):
    id: str
    key: str

class _DeploymentEndpointRecord(BaseModel):
    id: str
    endpoint: str

class _DeploymentSpec(BaseModel):
    tag: str
    name: str
    cpu: int | None = None
    gpu: DeploymentGPU | None = None
    gpu_count: int | None = None
    memory: int | None = None
    concurrency: int | None = None
    min_replicas: int | None = None
    max_replicas: int | None = None
    scaledown_window: float | None = None
    ssh_host: str | None = None      # --provider baremetal: full SSH target string
    endpoint_url: str | None = None  # --provider baremetal: public HTTP(S) base URL
    shared: bool = False
    pricing: _DeploymentPricing | None = None

class _Deployment(Protocol):
    @property
    def endpoint_url(self) -> str | None: ...
    @property
    def dashboard_url(self) -> str | None: ...
    def wait(self) -> None: ...  # block until live + healthy

class _DryRunDeployment:
    endpoint_url: str | None = None
    dashboard_url: str | None = None
    def wait(self) -> None:
        pass

class _BasetenDeployment:

    def __init__(self, deployment): # truss.api.definitions.ModelDeployment
        self._deployment = deployment

    @property
    def endpoint_url(self) -> str | None:
        return self._deployment._baseten_service.predict_url

    @property
    def dashboard_url(self) -> str | None:
        from truss.remote.baseten.service import URLConfig
        service = self._deployment._baseten_service
        return URLConfig.status_page_url(
            service._api.app_url,
            URLConfig.MODEL,
            service.model_id
        )

    def wait(self) -> None:
        self._deployment.wait_for_active()

class _ModalDeployment:

    def __init__(self, app, function): # modal.App, modal.Function
        self._app = app
        self._function = function

    @property
    def endpoint_url(self) -> str | None:
        base = self._function.get_web_url()
        return f"{base.rstrip('/')}/v1/chat/completions" if base else None

    @property
    def dashboard_url(self) -> str | None:
        return self._app.get_dashboard_url()

    def wait(self) -> None:
        base: str = self._function.get_web_url()
        if base is None:
            return
        health_url = f"{base.rstrip('/')}/health"
        deadline = time() + 45 * 60
        while time() < deadline:
            try:
                if get(health_url, timeout=10).status_code == 200:
                    return
            except Exception:
                pass
            sleep(5)

class _BaremetalDeployment:

    def __init__(self, endpoint_url: str, *, ssh_target: list[str]):
        self._base = endpoint_url.rstrip("/")
        self._ssh_target = ssh_target

    @property
    def endpoint_url(self) -> str | None:
        return f"{self._base}/v1/chat/completions"

    @property
    def dashboard_url(self) -> str | None:
        return None

    def wait(self) -> None:
        # First confirm the server came up on the node itself (over SSH); this also
        # tells us whether a public 404 means "not exposed" vs "still starting".
        node_healthy = self._wait_node_health()
        if not node_healthy:
            print(
                "[bold yellow]Warning:[/bold yellow] muna-server did not report healthy on the node "
                f"([bold]http://127.0.0.1:{_BAREMETAL_PORT}/health[/bold]) yet. It may still be starting; "
                "check [bold]/app/muna-server.log[/bold] on the node if the deployment does not come up."
            )
        health_url = f"{self._base}/health"
        deadline = time() + 45 * 60
        not_found_streak = 0
        while time() < deadline:
            try:
                status = get(health_url, timeout=10).status_code
                if status == 200:
                    return
                not_found_streak = not_found_streak + 1 if status == 404 else 0
            except Exception:
                not_found_streak = 0
            # If the server is healthy on the node but the public URL keeps returning 404,
            # the port almost certainly isn't exposed publicly; fail fast instead of waiting.
            if node_healthy and not_found_streak >= 6:
                print(
                    f"[bold red]Error:[/bold red] muna-server is healthy on the node but "
                    f"[bold cyan]{self._base}[/bold cyan] is not reachable (repeated 404s). "
                    f"Ensure port [bold]{_BAREMETAL_PORT}[/bold] is exposed at that URL. "
                    "On Runpod, add the port to the pod's [bold]Expose HTTP Ports[/bold] "
                    "(note: editing ports restarts the pod and changes the SSH port)."
                )
                raise Exit(code=1)
            sleep(5)

    def _wait_node_health(self, *, timeout: float = 120) -> bool:
        health_command = (
            f"curl -fsS -o /dev/null -w '%{{http_code}}' "
            f"http://127.0.0.1:{_BAREMETAL_PORT}/health"
        )
        deadline = time() + timeout
        while time() < deadline:
            result = run(
                ["ssh", *self._ssh_target, health_command],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip() == "200":
                return True
            sleep(5)
        return False

_BAREMETAL_PORT = 8000
_FXNC_VERSION = "0.0.46"
_MUNA_SERVER_VERSION = "0.0.2"
_TARGET_ARCH = "x86_64-unknown-linux-gnu"
_MUNA_SERVER_URL = (
    f"https://github.com/muna-ai/muna-server/releases/download/"
    f"{_MUNA_SERVER_VERSION}/muna-server-{_TARGET_ARCH}"
)
_FXNC_LIBRARY_URL = f"https://cdn.fxn.ai/fxnc/{_FXNC_VERSION}/libFunction-linux-x86_64.so"