#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pathlib import Path
from rich import print
from rich.table import Table
from typer import Argument, Exit, Option, Typer
from typing import Annotated

from ..beta.deployments import (
    DeploymentAccess, DeploymentPricing, DeploymentPricingKind,
    DurationDeploymentPricing, ImageDeploymentPricing,
    TokenDeploymentPricing
)
from ..logging import CustomProgress, CustomProgressTask
from ..muna import Muna
from .auth import get_access_key
from .compile import compile_predictor

app = Typer(no_args_is_help=True)

def deploy_function(
    target: Annotated[str, Argument(
        help="Predictor tag, or path to a Python module to compile and deploy.",
        metavar="TAG_OR_PATH"
    )],
    access: Annotated[DeploymentAccess, Option(
        help="Deployment access mode.",
        hidden=True
    )] = "private",
    pricing_kind: Annotated[DeploymentPricingKind | None, Option(
        "--pricing",
        hidden=True,
        help="Pricing kind for public endpoints."
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
    )] = None
):
    muna = Muna(get_access_key())
    pricing = _build_deployment_pricing(
        kind=pricing_kind,
        input_price=input_price,
        cached_input_price=cached_input_price,
        output_price=output_price
    )
    # Compile
    path = Path(target)
    if path.suffix == ".py" and path.is_file():
        tag = compile_predictor(path, muna=muna)
        print()
    elif target.startswith("@"):
        tag = target
    else:
        print(
            f"[bold red]Error:[/bold red] [bold cyan]{target}[/bold cyan] is neither "
            f"a predictor tag (e.g. [bold]@your-org/your-model[/bold]) nor a Python module."
        )
        raise Exit(code=1)
    # Deploy
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
        # Create deployment
        with CustomProgressTask(
            f"Deploying [bold cyan]{tag}[/bold cyan]...",
            done_text=f"Deployed [bold cyan]{tag}[/bold cyan]"
        ):
            deployment = muna.beta.deployments.create(
                tag,
                access=access,
                pricing=pricing
            )
    print(
        f"\n[bold spring_green3]🚀 {deployment.tag} is live.[/bold spring_green3] Call it at "
        f"[link={_INFERENCE_URL}][bold cyan]{_INFERENCE_URL}[/bold cyan][/link] "
        f"with model [bold cyan]{deployment.tag}[/bold cyan]."
    )

@app.command(name="list")
def list_deployments():
    """
    List your deployments.
    """
    muna = Muna(get_access_key())
    deployments = muna.beta.deployments.list()
    if not deployments:
        print("You have no deployments. Deploy a model with [bold orange1]muna deploy <tag>[/bold orange1].")
        return
    table = Table(box=None, header_style="bold", pad_edge=False)
    table.add_column("Tag", style="cyan")
    table.add_column("Access")
    table.add_column("Created", style="grey50")
    for deployment in deployments:
        table.add_row(
            deployment.tag,
            deployment.access,
            deployment.created
        )
    print(table)

@app.command(name="delete")
def delete_deployment(
    tag: Annotated[str, Argument(help="Predictor tag.")]
):
    """
    Delete a deployment.
    """
    muna = Muna(get_access_key())
    with CustomProgress():
        with CustomProgressTask(
            f"Deleting deployment of [bold cyan]{tag}[/bold cyan]...",
            done_text=f"Deleted deployment of [bold cyan]{tag}[/bold cyan]"
        ):
            deleted = muna.beta.deployments.delete(tag)
            if not deleted:
                print(f"[bold red]Error:[/bold red] [bold cyan]{tag}[/bold cyan] is not deployed.")
                raise Exit(code=1)

def _build_deployment_pricing(
    *,
    kind: DeploymentPricingKind | None,
    input_price: float | None,
    cached_input_price: float | None,
    output_price: float | None
) -> DeploymentPricing | None:
    match kind:
        case None:
            if any(value is not None for value in (input_price, cached_input_price, output_price)):
                print("[bold red]Error:[/bold red] Prices require [bold]--pricing[/bold].")
                raise Exit(code=1)
            return None
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
            return ImageDeploymentPricing(per_image=input_price)
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
            return DurationDeploymentPricing(per_minute=input_price)

_INFERENCE_URL = "https://inference.muna.ai/v1"