#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from rich import print
from typer import Argument, Exit, Option
from typing import Annotated, cast

from ..beta.deployments import (
    DeploymentGPU, DeploymentPricing, DeploymentPricingKind,
    DeploymentProvider, DurationDeploymentPricing,
    ImageDeploymentPricing, TokenDeploymentPricing
)
from ..logging import CustomProgress, CustomProgressTask
from ..muna import Muna
from ..types import Acceleration
from .auth import get_access_key

def deploy_function(
    tag: Annotated[
        str,
        Argument(help="Predictor tag.")
    ],
    provider: Annotated[
        DeploymentProvider,
        Option(help="Cloud to deploy the predictor to.")
    ],
    gpu: Annotated[
        DeploymentGPU | None,
        Option(help="GPU hardware configuration.")
    ] = None,
    name: Annotated[
        str | None,
        Option(help="Deployed model name.")
    ] = None,
    cpu: Annotated[int | None, Option(
        help="Number of vCPUs to request.",
        min=0,
    )] = None,
    gpu_count: Annotated[
        int | None,
        Option(help="Number of GPUs to request.", min=1)
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
        help=(
            "SSH target for --provider baremetal, e.g. "
            "'root@1.2.3.4 -p 22 -i ~/.ssh/key'. Accepts the full "
            "SSH target string (everything after `ssh`)."
        )
    )] = None,
    endpoint_url: Annotated[str | None, Option(
        help=(
            "Public HTTP(S) base URL where the deployed server is "
            "reachable. Required for --provider baremetal."
        )
    )] = None,
    shared: Annotated[bool, Option(
        "--shared",
        hidden=True,
        help="Create a shared endpoint. Requires superuser access."
    )] = False,
    pricing_kind: Annotated[DeploymentPricingKind | None, Option(
        "--pricing",
        hidden=True,
        help="Shared endpoint pricing kind."
    )] = None,
    input_price: Annotated[float | None, Option(
        "--input-price",
        hidden=True,
        min=0,
        help="Price per million input tokens, per image, or per minute in USD."
    )] = None,
    cached_input_price: Annotated[float | None, Option(
        "--cached-input-price",
        hidden=True,
        min=0,
        help="Price per million cached input tokens in USD."
    )] = None,
    output_price: Annotated[float | None, Option(
        "--output-price",
        hidden=True,
        min=0,
        help="Output price per million tokens in USD."
    )] = None,
    wait: Annotated[bool, Option(
        "--wait",
        help="Whether to wait until the deployment is complete."
    )] = False
):
    muna = Muna(get_access_key())
    with CustomProgress():
        # Retrieve predictor
        with CustomProgressTask(
            "Retrieving predictor...",
            done_text="Retrieved predictor"
        ):
            predictor = muna.predictors.retrieve(tag)
            if predictor is None:
                print(
                    f"[bold red]Error:[/bold red] Predictor "
                    f"[bold cyan]{tag}[/bold cyan] was not found or you do not "
                    "have access to it. Make sure you are signed in to the Muna "
                    "CLI with [bold orange1]muna auth login <access key>"
                    "[/bold orange1]."
                )
                raise Exit(code=1)
        # Parse pricing and acceleration
        pricing = _build_deployment_pricing(
            shared=shared,
            kind=pricing_kind,
            input_price=input_price,
            cached_input_price=cached_input_price,
            output_price=output_price
        )
        acceleration = cast(Acceleration, f"remote_{gpu or 'cpu'}")
        acceleration = (
            (acceleration, gpu_count)
            if gpu_count is not None
            else acceleration
        )
        try:
            # Create deployment
            with CustomProgressTask(
                f"Deploying predictor to {provider.title()}...",
                done_text=f"Deployed predictor to {provider.title()}"
            ):
                deployment = muna.beta.deployments.create(
                    tag,
                    provider=provider,
                    acceleration=acceleration,
                    name=name,
                    kind="shared" if shared else "dedicated",
                    cpu=cpu,
                    memory=memory,
                    concurrency=concurrency,
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    scaledown_window=scaledown_window,
                    ssh_host=ssh_host,
                    endpoint_url=endpoint_url,
                    pricing=pricing
                )
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"[bold red]Error:[/bold red] {error}")
            raise Exit(code=1)
        # Wait for muna-server to be healthy
        if wait:
            with CustomProgressTask(
                "Waiting for muna-server...",
                done_text="muna-server is healthy"
            ):
                deployment.wait()
    # Log dashboard URL
    if deployment.dashboard_url:
        print(
            f"Track deployment progress at "
            f"[link={deployment.dashboard_url}]"
            f"[bold cyan]{deployment.dashboard_url}[/bold cyan][/link]"
        )
    # Log endpoint URL
    if deployment.endpoint_url:
        print(
            f"Endpoint available at [link={deployment.endpoint_url}]"
            f"[bold cyan]{deployment.endpoint_url}[/bold cyan][/link]"
        )

def _build_deployment_pricing(
    *,
    shared: bool,
    kind: DeploymentPricingKind | None,
    input_price: float | None,
    cached_input_price: float | None,
    output_price: float | None
) -> DeploymentPricing | None:
    has_pricing_option = any(
        value is not None
        for value in (kind, input_price, cached_input_price, output_price)
    )
    if not shared:
        if has_pricing_option:
            print(
                "[bold red]Error:[/bold red] Pricing options require "
                "[bold]--shared[/bold]."
            )
            raise Exit(code=1)
        return None
    match kind:
        case None:
            print(
                "[bold red]Error:[/bold red] Shared endpoints require "
                "[bold]--pricing[/bold]."
            )
            raise Exit(code=1)
        case "tokens":
            if input_price is None:
                print(
                    "[bold red]Error:[/bold red] Token pricing requires "
                    "[bold]--input-price[/bold]."
                )
                raise Exit(code=1)
            return TokenDeploymentPricing(
                input_per_million=input_price,
                cached_input_per_million=cached_input_price,
                output_per_million=output_price
            )
        case "images":
            if output_price is not None:
                print(
                    "[bold red]Error:[/bold red] Image pricing does not "
                    "accept [bold]--output-price[/bold]."
                )
                raise Exit(code=1)
            if input_price is None:
                print(
                    "[bold red]Error:[/bold red] Image pricing requires "
                    "[bold]--input-price[/bold]."
                )
                raise Exit(code=1)
            return ImageDeploymentPricing(
                per_image=input_price
            )
        case "duration":
            if output_price is not None:
                print(
                    "[bold red]Error:[/bold red] Duration pricing does "
                    "not accept [bold]--output-price[/bold]."
                )
                raise Exit(code=1)
            if input_price is None:
                print(
                    "[bold red]Error:[/bold red] Duration pricing "
                    "requires [bold]--input-price[/bold]."
                )
                raise Exit(code=1)
            return DurationDeploymentPricing(
                per_minute=input_price
            )
