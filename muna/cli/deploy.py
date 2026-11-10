#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pathlib import Path
from rich import box, get_console, print
from rich.text import Text
from requests import RequestException
from time import monotonic
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from typer import Argument, Exit, Option, Typer
from typing import Annotated

from ..beta.deployments import (
    DeploymentError, DeploymentPricing, DeploymentPricingKind,
    DeploymentTenancy, DurationDeploymentPricing, ImageDeploymentPricing,
    TokenDeploymentPricing
)
from ..api import MunaAPIError
from ..beta.deployments.schema import parse_max_queue_time
from ..beta.deployments import DeploymentProgress
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
    tenancy: Annotated[DeploymentTenancy | None, Option(
        help="Share GPUs with other deployments."
    )] = None,
    max_queue_time: Annotated[str | None, Option(
        "--max-queue-time",
        help="Maximum time a request waits for compute before it is rejected, like '30s' or '10m' (at most 10 minutes).",
        metavar="DURATION"
    )] = None,
    listed: Annotated[bool | None, Option(
        "--listed/--unlisted",
        help="List a public predictor in the public catalog (superuser only).",
        hidden=True
    )] = None,
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
    # Parse before compiling, so that a typo doesn't waste a compile
    try:
        queue_time = parse_max_queue_time(max_queue_time) if max_queue_time is not None else None
    except ValueError as error:
        print(f"[bold red]Error:[/bold red] {error}.")
        raise Exit(code=1)
    pricing = _build_deployment_pricing(
        kind=pricing_kind,
        input_price=input_price,
        cached_input_price=cached_input_price,
        output_price=output_price
    )
    # Compile
    path = Path(target)
    if path.suffix == ".py" and path.is_file():
        try:
            tag = compile_predictor(
                path,
                muna=muna,
                required_targets=["x86_64-unknown-linux-gnu"],
                show_completion=False
            )
        except ValueError as error:
            print(f"[bold red]Error:[/bold red] {error}")
            raise Exit(code=1)
    elif target.startswith("@"):
        tag = target
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
    else:
        print(
            f"[bold red]Error:[/bold red] [bold cyan]{target}[/bold cyan] is neither "
            f"a predictor tag (e.g. [bold]@your-org/your-model[/bold]) nor a Python module."
        )
        raise Exit(code=1)
    # Compilation and predictor lookup finish before opening another Rich live view.
    console = get_console()
    started = monotonic()
    try:
        with _DeploymentProgressDisplay(console) as display:
            deployment = muna.beta.deployments.create(
                tag,
                listed=listed,
                tenancy=tenancy,
                max_queue_time=queue_time,
                pricing=pricing,
                progress=display.update
            )
    except KeyboardInterrupt:
        console.print(Text("\nStopped watching. Deployment continues in the background. Run muna deploy with the same tag and options to resume.", style="yellow"))
        raise Exit(130)
    except (DeploymentError, MunaAPIError, RequestException) as error:
        message = (
            "The progress connection was interrupted. Run muna deploy with the same tag and options to resume."
            if isinstance(error, RequestException) else str(error)
        )
        console.print(Text.assemble(("\n✗ ", "bold red"), message))
        raise Exit(1)
    console.print(Text.assemble(("  ✓ ", "bold green"), "Predictor deployed ", (f"in {monotonic() - started:.1f}s", "grey50")))
    if deployment.status and deployment.status.health.state == "unavailable":
        console.print(Text("Resources are prepared, but deployment health is currently unavailable. Retrieve the deployment for details.", style="yellow"))
        raise Exit(1)
    if deployment.status and deployment.status.health.state == "unknown":
        console.print(Text.assemble("Preparation is complete; current availability is unknown. Endpoint: ", (_INFERENCE_URL, "cyan")))
        return
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
    table = Table(
        box=box.ROUNDED,
        border_style="grey50",
        header_style="bold",
        show_lines=True
    )
    table.add_column("Tag", style="cyan")
    table.add_column("Tenancy")
    table.add_column("Max queue time")
    table.add_column("Created", style="grey50")
    for deployment in deployments:
        table.add_row(
            deployment.tag,
            deployment.tenancy,
            f"{deployment.max_queue_time.total_seconds():g}s" if deployment.max_queue_time is not None else "Default",
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

class _DeploymentProgressDisplay:
    """
    One preparation attempt, with a readable transcript outside a terminal.
    """

    def __init__(self, console: Console):
        self.console = console
        self.progress = DeploymentProgress(stage="preparing")
        self.started = monotonic()
        self.spinner = Spinner("dots", style="cyan")
        self.live: Live | None = None
        self.previous: tuple[str, int | None, bool] | None = None

    def __enter__(self):
        if self.console.is_terminal:
            self.live = Live(
                self,
                console=self.console,
                refresh_per_second=12,
                transient=True
            )
            self.live.__enter__()
        else:
            self.update(self.progress)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.live:
            self.live.__exit__(exc_type, exc, traceback)

    @property
    def label(self) -> str:
        if self.progress.retrying:
            return "Retrying deployment"
        return {
            "preparing": "Deploying predictor",
            "resolving": "Deploying predictor",
            "waiting": "Waiting for a serving node",
            "downloading": "Downloading resources"
        }[self.progress.stage]

    def update(self, progress: DeploymentProgress):
        self.progress = progress
        if self.console.is_terminal:
            return
        bucket = None if progress.fraction is None else int(progress.fraction * 10)
        key = (self.label, bucket, progress.retrying)
        if key != self.previous:
            line = Text.assemble(("  • ", "yellow" if progress.retrying else "cyan"), f"{self.label}…")
            if progress.fraction is not None:
                line.append(f" {int(progress.fraction * 100)}%", style="bold")
            self.console.print(line)
            self.previous = key

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions
    ) -> RenderResult:
        color = "yellow" if self.progress.retrying else "cyan"
        self.spinner.style = color
        line = Text(f"{self.label}…", no_wrap=True, overflow="ellipsis")
        elapsed = f"  {int(monotonic() - self.started)}s"
        fraction = self.progress.fraction
        width = min(24, max(0, options.max_width - len(self.label) - len(elapsed) - 14))
        if fraction is not None:
            if width >= 6:
                filled = int(fraction * width)
                line.append("  ")
                line.append("━" * filled, style=color)
                line.append("─" * (width - filled), style="grey35")
            line.append(f" {int(fraction * 100):3d}%", style="bold")
        line.append(elapsed, style="grey50")
        row = Table.grid(padding=(0, 1))
        row.add_column(width=1)
        row.add_column(overflow="ellipsis", no_wrap=True)
        row.add_row(self.spinner, line)
        yield row

_INFERENCE_URL = "https://inference.muna.ai/v1"