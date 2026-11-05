# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmembers, getmodulename, isfunction
from pathlib import Path
from pydantic import BaseModel
from rich import get_console, print as print_rich
from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
import sys
from time import monotonic
from typer import Argument, Exit, Option
from typing import Annotated, Callable, Iterator, Literal
from urllib.parse import urlparse, urlunparse

from ..api import MunaAPIError
from ..beta.compile import CompileDialect
from ..compile import PredictorSpec
from ..logging import CustomProgress, CustomProgressTask
from ..muna import Muna
from ..sandbox import EntrypointCommand
from .auth import get_access_key

def compile_function(
    path: Annotated[str, Argument(
        resolve_path=True,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        help="Python source path or URL."
    )]
):
    muna = Muna(get_access_key())
    compile_predictor(Path(path), muna=muna)

def compile_predictor(
    path: Path,
    *,
    muna: Muna
) -> str:
    """
    Compile the predictor function in a Python module, showing live progress.
    Raises `CompileError` if the predictor fails to compile.

    Returns:
        str: Predictor tag.
    """
    path = path.resolve()
    with CustomProgress():
        # Load
        with CustomProgressTask(loading_text="Loading predictor...") as task:
            func = _load_predictor_func(path)
            entrypoint = EntrypointCommand(
                from_path=str(path),
                to_path=f"./{path.name}",
                name=func.__name__
            )
            spec: PredictorSpec = func.__predictor_spec
            if spec.tag is None:
                user = muna.users.retrieve()
                spec.tag = f"@{user.username}/{func.__name__.replace('_', '-')}"
            if spec.description is None:
                if func.__doc__ is None:
                    raise ValueError(
                        f"Cannot compile predictor because no `description` was "
                        f"provided in `@compile(...)`, and the function does not "
                        f"have a docstring."
                    )
                spec.description = func.__doc__.strip()
            task.finish(f"Loaded Python function: [bold cyan]{spec.tag}[/bold cyan]")
        # Populate sandbox
        sandbox = spec.sandbox
        sandbox.commands.append(entrypoint)
        with CustomProgressTask(
            "Uploading sandbox...",
            done_text="Uploaded sandbox"
        ):
            sandbox.populate(muna=muna)
        # Populate dialects
        if spec.dialects:
            spec.dialects = [_populate_dialect(d, muna=muna) for d in spec.dialects]
        # Compile
        with CustomProgressTask(
            "Running codegen...",
            done_text="Completed codegen"
        ):
            with CustomProgressTask("Creating predictor..."):
                try:
                    muna.client.request(
                        method="POST",
                        path="/predictors",
                        body=spec.model_dump(
                            mode="json",
                            exclude=spec.model_extra.keys(),
                            exclude_none=True,
                            by_alias=True
                        ),
                        response_type=_Predictor
                    )
                except MunaAPIError as ex:
                    if ex.status_code != 409:
                        raise
            events = muna.client.stream(
                method="POST",
                path=f"/predictors/{spec.tag}/compile",
                body=spec.model_dump(
                    mode="json",
                    exclude=spec.model_extra.keys(),
                    by_alias=True
                ),
                response_type=_LogEvent | _ErrorEvent | _CompileEvent
            )
            first_compile_event: _CompileEvent | None = None
            with ProgressLogQueue() as task_queue:
                for event in events:
                    match event.event:
                        case "log":
                            task_queue.push_log(event)
                        case "error":
                            task_queue.push_error(event)
                            raise CompileError(event.data.error)
                        case "compile":
                            # Leave the codegen view, since Rich cannot run two live displays
                            # at once. The rest of the stream is consumed by the compile view.
                            first_compile_event = event
                            break
    predictor_url = _compute_predictor_url(muna.client.api_url, spec.tag)
    # The API predates compile progress, so implementations compile without a live view
    if first_compile_event is None:
        print_rich(
            f"\n[bold spring_green3]🎉 Predictor is now being compiled.[/bold spring_green3] "
            f"Check it out at [link={predictor_url}]{predictor_url}[/link]"
        )
        return spec.tag
    _watch_compile(
        first_compile_event,
        events=events,
        tag=spec.tag,
        predictor_url=predictor_url
    )
    # Return
    return spec.tag

def _watch_compile(
    first_event: _CompileEvent,
    *,
    events: Iterator[_LogEvent | _ErrorEvent | _CompileEvent],
    tag: str,
    predictor_url: str
):
    console = get_console()
    view = _CompileView(first_event.data.targets)
    try:
        with Live(view, console=console, refresh_per_second=12):
            for event in events:
                match event.event:
                    case "compile":
                        view.targets = event.data.targets
                    case "error":
                        view.final = True
                        raise CompileError(event.data.error)
            view.final = True
    except KeyboardInterrupt:
        console.print(
            f"\n[bold yellow]Stopped watching.[/bold yellow] Compilation continues in the "
            f"background. Track it at [link={predictor_url}]{predictor_url}[/link]"
        )
        raise Exit(130)
    # Report
    elapsed = _format_duration(monotonic() - view.start)
    targets = view.targets
    console.print()
    if not all(counts.is_done for counts in targets.values()):
        console.print(
            f"[bold yellow]Still compiling after {elapsed}.[/bold yellow] Compilation continues "
            f"in the background. Track it at [link={predictor_url}]{predictor_url}[/link]"
        )
        return
    failed_targets = [target for target, counts in targets.items() if counts.active == 0]
    if failed_targets:
        raise CompileError(
            f"Failed to compile predictor for {', '.join(failed_targets)} because no "
            f"implementation compiled for {'this target' if len(failed_targets) == 1 else 'these targets'}. "
            f"See the compile logs at {predictor_url}"
        )
    console.print(
        f"[bold spring_green3]🎉 Compiled {tag} in {elapsed}.[/bold spring_green3] "
        f"Check it out at [link={predictor_url}]{predictor_url}[/link]"
    )
    failed = sum(counts.error for counts in targets.values())
    if failed:
        console.print(
            f"[grey50]{failed} implementation{'s' if failed != 1 else ''} failed to compile; "
            f"other implementations cover every target.[/grey50]"
        )

def _populate_dialect(
    dialect: CompileDialect,
    *,
    muna: Muna
) -> CompileDialect:
    if dialect.kind == "builtin":
        return dialect
    if dialect.url is not None:
        return dialect
    label = _describe_dialect(dialect)
    with CustomProgressTask(
        f"Uploading dialect [yellow2]{label}[/yellow2]...",
        done_text=f"Uploaded dialect [yellow2]{label}[/yellow2]"
    ):
        return dialect.populate(muna=muna)

def _describe_dialect(dialect: CompileDialect) -> str:
    if dialect._path is not None:
        try:
            return str(dialect._path.relative_to(Path.cwd()))
        except ValueError:
            return str(dialect._path)
    if dialect._git_url is not None:
        if dialect._git_revision:
            return f"{dialect._git_url}@{dialect._git_revision}"
        return dialect._git_url
    return "<unknown>"

def _load_predictor_func(path: str) -> Callable[...,object]:
    if "" not in sys.path:
        sys.path.insert(0, "")
    path: Path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"Cannot compile predictor because no Python module exists at the given path.")
    sys.path.insert(0, str(path.parent))
    name = getmodulename(path)
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    main_func = next(func for _, func in getmembers(module, isfunction) if hasattr(func, "__predictor_spec"))
    return main_func

def _compute_predictor_url(
    api_url: str,
    tag: str
) -> str:
    parsed_url = urlparse(api_url)
    hostname_parts = parsed_url.hostname.split(".")
    if hostname_parts[0] == "api":
        hostname_parts.pop(0)
    hostname = ".".join(hostname_parts)
    netloc = hostname if not parsed_url.port else f"{hostname}:{parsed_url.port}"
    predictor_url = urlunparse(parsed_url._replace(netloc=netloc, path=f"{tag}"))
    return predictor_url

class _Predictor(BaseModel):
    tag: str

class _LogData(BaseModel):
    message: str
    level: int = 0
    status: Literal["success", "error"] = "success"
    update: bool = False

class _LogEvent(BaseModel):
    event: Literal["log"]
    data: _LogData

class _ErrorData(BaseModel):
    error: str

class _ErrorEvent(BaseModel):
    event: Literal["error"]
    data: _ErrorData

class _TargetCounts(BaseModel):
    idle: int = 0
    compiling: int = 0
    active: int = 0
    error: int = 0

    @property
    def total(self) -> int:
        return self.idle + self.compiling + self.active + self.error

    @property
    def is_done(self) -> bool:
        return self.idle == 0 and self.compiling == 0

class _CompileData(BaseModel):
    targets: dict[str, _TargetCounts]

class _CompileEvent(BaseModel):
    event: Literal["compile"]
    data: _CompileData

class CompileError(Exception):
    pass

class _CompileView:
    """
    Live per-target view of implementation compiles.
    Each `compile` event carries a full snapshot, so the view only renders the latest one.
    """

    def __init__(self, targets: dict[str, _TargetCounts]):
        self.targets = targets
        self.final = False
        self.start = monotonic()
        # Spinners animate relative to their first render, so they must persist across renders
        self.__spinner = Spinner("dots")
        self.__target_spinners: dict[str, Spinner] = { }

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions
    ) -> RenderResult:
        targets = self.targets
        total = sum(counts.total for counts in targets.values())
        all_done = all(counts.is_done for counts in targets.values())
        any_failed = any(counts.is_done and counts.active == 0 for counts in targets.values())
        if not all_done:
            header_icon = self.__pending_icon(self.__spinner)
        elif any_failed:
            header_icon = Text("✘", style="bright_red")
        else:
            header_icon = Text("✔", style="bold green")
        header = Table.grid(padding=(0, 1))
        header.add_column(width=1)
        header.add_column()
        header.add_row(
            header_icon,
            Text.assemble(
                f"Compiling {total} implementations ",
                (_format_duration(monotonic() - self.start), "grey50")
            )
        )
        # The bar only fits next to the longest target triple on wide terminals
        show_bar = options.max_width >= 100
        rows = Table.grid(padding=(0, 1))
        rows.add_column(no_wrap=True, min_width=3)
        rows.add_column(no_wrap=True, min_width=1)
        rows.add_column(no_wrap=True)
        if show_bar:
            rows.add_column(no_wrap=True)
        rows.add_column(no_wrap=True)
        for target, counts in targets.items():
            spinner = self.__target_spinners.setdefault(target, Spinner("dots"))
            cells = [
                Text("└──", style="grey50"),
                self.__target_icon(counts, spinner),
                Text(target, style="cyan")
            ]
            if show_bar:
                cells.append(_render_bar(counts, width=16))
            cells.append(_render_summary(counts))
            rows.add_row(*cells)
        yield Group(header, rows)

    def __target_icon(
        self,
        counts: _TargetCounts,
        spinner: Spinner
    ):
        if not counts.is_done:
            return self.__pending_icon(spinner)
        if counts.active == 0:
            return Text("✘", style="bright_red")
        return Text("✔", style="bold green")

    def __pending_icon(self, spinner: Spinner):
        return Text("…", style="grey50") if self.final else spinner

def _render_bar(
    counts: _TargetCounts,
    *,
    width: int
) -> Text:
    total = max(counts.total, 1)
    segments = [
        (counts.active, "green"),
        (counts.error, "bright_red"),
        (counts.compiling, "yellow"),
        (counts.idle, "grey35"),
    ]
    text = Text()
    used = 0
    for i, (count, style) in enumerate(segments):
        n = round(count / total * width) if i < len(segments) - 1 else width - used
        n = max(0, min(n, width - used))
        text.append("━" * n, style=style)
        used += n
    return text

def _render_summary(counts: _TargetCounts) -> Text:
    text = Text(f"{counts.active}/{counts.total} compiled", style="bold" if counts.is_done else "")
    if counts.compiling:
        text.append(f" · {counts.compiling} compiling", style="yellow")
    if counts.idle:
        text.append(f" · {counts.idle} queued", style="grey50")
    if counts.error:
        text.append(f" · {counts.error} failed", style="bright_red")
    return text

def _format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"

class ProgressLogQueue:

    def __init__(self):
        self.queue: list[tuple[int, CustomProgressTask]] = []

    def push_log(self, event: _LogEvent):
        # Check for update
        if event.data.update and self.queue:
            current_level, current_task = self.queue[-1]
            current_task.update(description=event.data.message, status=event.data.status)
            return
        # Pop
        while self.queue:
            current_level, current_task = self.queue[-1]
            if event.data.level > current_level:
                break
            current_task.__exit__(None, None, None)
            self.queue.pop()
        task = CustomProgressTask(event.data.message)
        task.__enter__()
        self.queue.append((event.data.level, task))

    def push_error(self, error: _ErrorEvent):
        while self.queue:
            _, current_task = self.queue.pop()
            current_task.__exit__(RuntimeError, None, None)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        while self.queue:
            _, current_task = self.queue.pop()
            current_task.__exit__(None, None, None)