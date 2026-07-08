# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel
from requests import get
from rich import print
from shlex import split
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
        endpoint_url=endpoint_url
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
    truss_config = _build_truss_config(spec)
    if dry_run:
        predictor_slug = spec.tag.lstrip("@").replace("/", "_")
        directory = Path(f"{predictor_slug}-truss")
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "config.yaml"
        truss_config.write_to_yaml_file(config_path)
        print(f"Wrote Truss config to [bold cyan]{config_path}[/bold cyan]")
        return _DryRunDeployment()
    if "baseten" not in RemoteFactory.get_available_config_names():
        print(
            "[bold red]Error:[/bold red] No Baseten remote is configured. "
            "Run [bold]truss login[/bold] to authenticate with Baseten."
        )
        raise Exit(code=1)
    with TemporaryDirectory() as directory:
        truss_config.write_to_yaml_file(Path(directory) / "config.yaml")
        deployment = push(directory, remote="baseten", publish=True)
    return _BasetenDeployment(deployment)

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
            Secret.from_dict({ "MUNA_ACCESS_KEY": muna.client.access_key })
        ],
        timeout=60 * 60,
        startup_timeout=45 * 60,
        scaledown_window=spec.scaledown_window,
        serialized=True
    )
    @web_server(8000, startup_timeout=45 * 60)
    def serve():
        process = Popen(["/app/muna-server"])
    with enable_output():
        app.deploy()
    return _ModalDeployment(app, serve)

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
    # Build the remote setup + launch script
    script = _build_baremetal_script(
        tag=spec.tag,
        access_key=muna.client.access_key
    )
    if dry_run:
        print(script)
        return _DryRunDeployment()
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
    return _BaremetalDeployment(spec.endpoint_url, ssh_target=ssh_target)

def _build_baremetal_script(*, tag: str, access_key: str) -> str:
    return (
        f"set -e\n"
        f"export DIR=/app\n"
        f"mkdir -p \"$DIR\"\n"
        f"curl -fsSL {_MUNA_SERVER_URL} -o \"$DIR/muna-server\" && chmod +x \"$DIR/muna-server\"\n"
        f"curl -fsSL {_FXNC_LIBRARY_URL} -o \"$DIR/libFunction.so\"\n"
        f"# preload weights up front (download only) so the first request skips the download;\n"
        f"# uses the same MUNA_HOME as serve so the cache is shared.\n"
        f"env LD_LIBRARY_PATH=\"$DIR\" MUNA_HOME=\"$DIR/.muna\" MUNA_ACCESS_KEY={access_key} "
        f"\"$DIR/muna-server\" preload \"{tag}\"\n"
        f"# stop a previous instance if present\n"
        f"[ -f \"$DIR/muna-server.pid\" ] && kill \"$(cat \"$DIR/muna-server.pid\")\" 2>/dev/null || true\n"
        f"# detach; `exec` keeps muna-server on the same PID we record\n"
        f"setsid bash -c 'echo $$ > \"$DIR/muna-server.pid\"; "
        f"exec env LD_LIBRARY_PATH=\"$DIR\" MUNA_HOME=\"$DIR/.muna\" MUNA_ACCESS_KEY={access_key} "
        f"PORT={_BAREMETAL_PORT} \"$DIR/muna-server\" serve' "
        f"> \"$DIR/muna-server.log\" 2>&1 </dev/null &\n"
    )

def _build_truss_config(spec: _DeploymentSpec):
    from truss.base.truss_config import (
        AcceleratorSpec, BaseImage, DockerServer,
        Resources, Runtime, TrussConfig
    )
    START_COMMAND = (
        'sh -c "export PORT=8000 LD_LIBRARY_PATH=/app/data MUNA_HOME=/app/.fxn '
        'MUNA_ACCESS_KEY=$(cat /secrets/MUNA_ACCESS_KEY); '
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
        secrets={ "MUNA_ACCESS_KEY": None },
        resources=resources,
    )
    if spec.concurrency is not None:
        config.runtime = Runtime(predict_concurrency=spec.concurrency)
    return config

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

class _Deployment(Protocol):
    endpoint_url: str | None     # OpenAI-compatible inference endpoint; None when not-yet-live / dry-run
    dashboard_url: str | None    # provider console page (build progress, logs, status); None for dry-run
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