#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import PrivateAttr
from requests import get
from requests.exceptions import RequestException
from rich import print
from shlex import quote
from subprocess import run
from time import sleep, time

from ._runtime import FXNC_LIBRARY_URL, MUNA_SERVER_URL, SERVER_PORT
from ._schema import Deployment

class SSHDeploymentError(RuntimeError):
    pass

class SSHDeployment(Deployment):
    _base: str = PrivateAttr()
    _ssh_target: list[str] = PrivateAttr()
    _node_port: int = PrivateAttr()
    _poll_interval: float = PrivateAttr()
    _health_timeout: float = PrivateAttr()
    _node_health_timeout: float = PrivateAttr()
    _node_log_path: str = PrivateAttr()
    _not_found_limit: int | None = PrivateAttr()

    def __init__(
        self,
        base_url: str,
        *,
        ssh_target: list[str],
        dashboard_url: str | None = None,
        node_port: int = SERVER_PORT,
        poll_interval: float = 5,
        health_timeout: float = 45 * 60,
        node_health_timeout: float = 120,
        node_log_path: str = "/app/muna-server.log",
        not_found_limit: int | None = None
    ):
        base = base_url.rstrip("/")
        super().__init__(
            endpoint_url=f"{base}/v1",
            dashboard_url=dashboard_url
        )
        self._base = base
        self._ssh_target = list(ssh_target)
        self._node_port = node_port
        self._poll_interval = poll_interval
        self._health_timeout = health_timeout
        self._node_health_timeout = node_health_timeout
        self._node_log_path = node_log_path
        self._not_found_limit = not_found_limit

    def wait(self) -> None:
        # Wait until `muna-server` is healthy on the node
        node_healthy = self._wait_node_health()
        if not node_healthy:
            print(
                "[bold yellow]Warning:[/bold yellow] muna-server has not reported healthy on the node "
                f"([bold]http://127.0.0.1:{self._node_port}/health[/bold]). "
                "It may still be starting; check "
                f"[bold]{self._node_log_path}[/bold] on the node if the deployment does not come up."
            )
        # Loop until `muna-server` is reachable over TCP
        deadline = time() + self._health_timeout
        not_found_streak = 0
        while time() < deadline:
            try:
                status = get(
                    f"{self._base}/health",
                    timeout=10
                ).status_code
                if status == 200:
                    return
                not_found_streak = (
                    not_found_streak + 1
                    if status == 404
                    else 0
                )
            except RequestException:
                not_found_streak = 0
            # Not reachable
            if (
                node_healthy
                and self._not_found_limit is not None
                and not_found_streak >= self._not_found_limit
            ):
                message = (
                    f"muna-server is healthy on the node but {self._base} is not reachable. "
                    f"Ensure port {self._node_port} is exposed at that URL."
                )
                print(f"[bold red]Error:[/bold red] {message}")
                raise SSHDeploymentError(message)
            # Wait
            sleep(self._poll_interval)
        # Fail
        raise SSHDeploymentError("Timed out waiting for the public endpoint to become healthy.")

    def _wait_node_health(self) -> bool:
        """
        Wait until muna-server reports healthy over SSH.
        """
        health_command = (
            "curl -fsS -o /dev/null -w '%{http_code}' "
            f"http://127.0.0.1:{self._node_port}/health"
        )
        deadline = time() + self._node_health_timeout
        while time() < deadline:
            result = run(
                ["ssh", *self._ssh_target, health_command],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip() == "200":
                return True
            sleep(self._poll_interval)
        return False

def build_server_script(
    *,
    tag: str,
    access_key: str,
    install_dir: str = "/app"
) -> str:
    """
    Build a remote script that installs, preloads, and launches muna-server.
    """
    access_key = quote(access_key)
    tag = quote(tag)
    serve_command = (
        'echo $$ > "$DIR/muna-server.pid"; '
        'exec env LD_LIBRARY_PATH="$DIR" MUNA_HOME="$DIR/.muna" '
        f"MUNA_ACCESS_KEY={access_key} MUNA_SERVER_MODELS={tag} PORT={SERVER_PORT} "
        '"$DIR/muna-server" serve'
    )
    return (
        "set -e\n"
        f"export DIR={install_dir}\n"
        'mkdir -p "$DIR"\n'
        f'curl -fsSL {MUNA_SERVER_URL} -o "$DIR/muna-server" && '
        'chmod +x "$DIR/muna-server"\n'
        f'curl -fsSL {FXNC_LIBRARY_URL} -o "$DIR/libFunction.so"\n'
        'env LD_LIBRARY_PATH="$DIR" MUNA_HOME="$DIR/.muna" '
        f"MUNA_ACCESS_KEY={access_key} "
        f'"$DIR/muna-server" preload {tag}\n'
        '[ -f "$DIR/muna-server.pid" ] && '
        'kill "$(cat "$DIR/muna-server.pid")" 2>/dev/null || true\n'
        f"setsid bash -c {quote(serve_command)} "
        '> "$DIR/muna-server.log" 2>&1 </dev/null &\n'
    )
