# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import PrivateAttr
from requests import get
from subprocess import Popen
from sys import version_info
from time import sleep, time
from typing import cast, Any

from ._runtime import FXNC_LIBRARY_URL, MUNA_SERVER_URL, SERVER_PORT
from ._schema import Deployment, DeploymentSpec

class ModalDeployment(Deployment):
    """
    Deployment on Modal.
    """
    _function: Any = PrivateAttr()

    def __init__(self, app, function):
        from modal import App, Function
        app = cast(App, app)
        function = cast(Function, function)
        super().__init__(
            endpoint_url=f"{function.get_web_url().rstrip('/')}/v1",
            dashboard_url=app.get_dashboard_url()
        )
        self._function = function

    def wait(self) -> None:
        from modal import Function
        function = cast(Function, self._function)
        base = function.get_web_url()
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

def create_modal_deployment(
    spec: DeploymentSpec,
    *,
    deployment_key: str
) -> ModalDeployment:
    """
    Deploy a predictor to Modal.
    """
    _validate_modal_deployment()
    # Import Modal
    from modal import (
        enable_output, web_server, App,
        Image, Secret, Volume
    )
    # Create Modal app and image
    predictor_slug = (
        spec.tag
        .lstrip("@")
        .replace("/", "_")
    )
    app = App(f"muna-{predictor_slug}")
    volume = Volume.from_name(
        "muna-deploy-cache",
        create_if_missing=True,
        version=2
    )
    image = (
        Image
        .debian_slim(f"{version_info.major}.{version_info.minor}")
        .apt_install("curl")
        .run_commands(
            "mkdir -p /app && "
            f"curl -fsSL {MUNA_SERVER_URL} "
            "-o /app/muna-server && "
            "chmod +x /app/muna-server && "
            f"curl -fsSL {FXNC_LIBRARY_URL} "
            "-o /app/libFunction.so"
        )
    )
    # Define server
    @app.function(
        image=image,
        cpu=spec.cpu,
        gpu=(
            f"{spec.gpu}:{spec.gpu_count or 1}"
            if spec.gpu is not None
            else None
        ),
        memory=spec.memory,
        min_containers=spec.min_replicas,
        max_containers=spec.max_replicas,
        volumes={"/muna": volume},
        env={
            "LD_LIBRARY_PATH": "/app",
            "MUNA_HOME": "/muna"
        },
        secrets=[
            Secret.from_dict({
                "MUNA_ACCESS_KEY": deployment_key
            })
        ],
        timeout=60 * 60,
        startup_timeout=45 * 60,
        scaledown_window=spec.scaledown_window,
        serialized=True
    )
    @web_server(
        SERVER_PORT,
        startup_timeout=45 * 60
    )
    def serve():
        Popen(["/app/muna-server"])
    # Deploy
    with enable_output():
        app.deploy()
    # Return
    return ModalDeployment(app, serve)

def _validate_modal_deployment() -> None:
    """
    Validate the Modal environment before creating an API record.
    """
    try:
        from modal import (
            enable_output as _, web_server as _,
            App as _, Image as _, Secret as _, Volume as _
        )
    except ImportError as error:
        raise ImportError(
            "The `modal` package is required for Modal deployments. "
            "Install it with `pip install modal`."
        ) from error