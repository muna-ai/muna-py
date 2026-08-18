# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from shlex import split
from subprocess import run
from warnings import warn

from ._runtime import SERVER_PORT
from ._schema import DeploymentSpec
from ._ssh import build_server_script, SSHDeployment

class BaremetalDeployment(SSHDeployment):
    """
    Deployment on a baremetal node (over SSH).
    """

    def __init__(
        self,
        endpoint_url: str,
        *,
        ssh_target: list[str]
    ):
        super().__init__(
            endpoint_url,
            ssh_target=ssh_target,
            node_port=SERVER_PORT,
            node_log_path="/app/muna-server.log",
            not_found_limit=6
        )

def create_baremetal_deployment(
    spec: DeploymentSpec,
    *,
    deployment_key: str
) -> BaremetalDeployment:
    """
    Deploy a predictor to a baremetal node over SSH.
    """
    # Check that SSH host is provided
    if spec.ssh_host is None:
        raise ValueError("ssh_host is required for baremetal deployments.")
    # Check that endpoint URL is provided
    if spec.endpoint_url is None:
        raise ValueError("endpoint_url is required for baremetal deployments.")
    # Inform user of ignored args
    ignored = [
        name
        for name, value in {
            "cpu": spec.cpu,
            "gpu": spec.gpu,
            "gpu_count": spec.gpu_count,
            "memory": spec.memory,
            "concurrency": spec.concurrency,
            "min_replicas": spec.min_replicas,
            "max_replicas": spec.max_replicas,
            "scaledown_window": spec.scaledown_window
        }.items()
        if value is not None
    ]
    if ignored:
        warn(
            "Ignoring resource and autoscaling options that do not apply "
            f"to a fixed baremetal node: {', '.join(ignored)}.",
            stacklevel=2
        )
    # Build server script
    ssh_target = split(spec.ssh_host)
    script = build_server_script(
        tag=spec.tag,
        access_key=deployment_key
    )
    # Run deployment command
    result = run(
        ["ssh", *ssh_target, "bash -s"],
        input=script,
        capture_output=True,
        text=True
    )
    # Check
    if result.returncode != 0:
        detail = result.stderr.strip()
        message = (
            "Failed to install and launch muna-server on the node "
            f"(SSH exited with code {result.returncode})."
        )
        raise RuntimeError(
            f"{message} {detail}" if detail else message
        )
    # Return
    return BaremetalDeployment(
        spec.endpoint_url,
        ssh_target=ssh_target
    )