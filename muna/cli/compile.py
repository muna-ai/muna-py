# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from base64 import b64decode
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getmembers, getmodulename, isfunction
from os import environ
from pathlib import Path
import platform
from pydantic import BaseModel
import re
from requests import get as http_get
from rich import print as print_rich
import sys
from tempfile import TemporaryDirectory
from typer import Argument, Option
from typing import Annotated, Callable, Literal
from urllib.parse import urlparse, urlunparse

from ..client import MunaAPIError
from ..compile import PredictorSpec
from ..logging import CustomProgress, CustomProgressTask
from ..muna import Muna
from ..sandbox import EntrypointCommand
from ..types import PredictionResource
from .auth import get_access_key

def compile_function(
    path: Annotated[str, Argument(
        resolve_path=True,
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        help="Python source path."
    )],
    overwrite: Annotated[bool, Option(
        "--overwrite",
        help="Whether to delete any existing predictor with the same tag before compiling.")
    ]=False,
):
    muna = Muna(get_access_key())
    path: Path = Path(path).resolve()
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
                spec.tag = f"@{user.username}/{func.__name__}"
            if spec.description is None:
                if func.__doc__ is None:
                    raise ValueError(
                        f"Cannot compile predictor because no `description` was "
                        f"provided in `@compile(...)`, and the function does not "
                        f"have a docstring."
                    )
                spec.description = func.__doc__.strip()
            task.finish(f"Loaded Python function: [bold cyan]{spec.tag}[/bold cyan]")
        # Populate
        sandbox = spec.sandbox
        sandbox.commands.append(entrypoint)
        with CustomProgressTask(loading_text="Uploading sandbox...", done_text="Uploaded sandbox"):
            sandbox.populate(muna=muna)
        # Compile
        with CustomProgressTask(loading_text="Running codegen...", done_text="Completed codegen"):
            with CustomProgressTask(loading_text="Creating predictor..."):
                try:
                    muna.client.request(
                        method="POST",
                        path="/predictors",
                        body=spec.model_dump(
                            mode="json",
                            exclude=spec.model_extra.keys(),
                            by_alias=True
                        ),
                        response_type=_Predictor
                    )
                except MunaAPIError as ex:
                    if ex.status_code != 409 or not overwrite:
                        raise
            with ProgressLogQueue() as task_queue:
                for event in muna.client.stream(
                    method="POST",
                    path=f"/predictors/{spec.tag}/compile",
                    body=spec.model_dump(
                        mode="json",
                        exclude=spec.model_extra.keys(),
                        by_alias=True
                    ),
                    response_type=_LogEvent | _ErrorEvent
                ):
                    match event.event:
                        case "log":
                            task_queue.push_log(event)
                        case "error":
                            task_queue.push_error(event)
                            raise CompileError(event.data.error)
    predictor_url = _compute_predictor_url(muna.client.api_url, spec.tag)
    print_rich(f"\n[bold spring_green3]🎉 Predictor is now being compiled.[/bold spring_green3] Check it out at [link={predictor_url}]{predictor_url}[/link]")

def transpile_function(
    path: Annotated[
        str,
        Argument(help="Python source path or GitHub URL.")
    ],
    output: Annotated[Path, Option(
        resolve_path=True,
        exists=False,
        writable=True,
        help="Output path for generated C++ sources."
    )]=Path("cpp"),
    trust_remote_code: Annotated[bool, Option(
        "--trust-remote-code",
        help="Trust and execute code from remote URLs. Required when using GitHub URLs."
    )]=False,
):
    muna = Muna(get_access_key())
    # Resolve path
    path: Path = _resolve_source_path(path, trust_remote_code=trust_remote_code)
    # Check output
    if output.exists():
        raise ValueError(f"Cannot transpile because output directory already exists: {output}")
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
            spec.targets = (
                spec.targets
                if spec.targets is not None
                else [_get_current_target()]
            )
            task.finish(f"Loaded Python function: [bold cyan]{func.__module__}.{func.__name__}[/bold cyan]")
        # Populate
        sandbox = spec.sandbox
        sandbox.commands.append(entrypoint)
        with CustomProgressTask(loading_text="Uploading sandbox...", done_text="Uploaded sandbox"):
            sandbox.populate(muna=muna)
        # Compile
        with CustomProgressTask(loading_text="Running codegen...", done_text="Completed codegen"):
            with ProgressLogQueue() as task_queue:
                for event in muna.client.stream(
                    method="POST",
                    path=f"/transpile",
                    body=spec.model_dump(
                        mode="json",
                        exclude=spec.model_extra.keys(),
                        by_alias=True
                    ),
                    response_type=_LogEvent | _ErrorEvent | _SourceEvent
                ):
                    match event.event:
                        case "log":
                            task_queue.push_log(event)
                        case "error":
                            task_queue.push_error(event)
                            raise CompileError(event.data.error)
                        case "sources":
                            source: _TranspiledSource = event.data[0]
        # Write source files
        output.mkdir()
        _write_file(source.code, dir=output, muna=muna)
        _write_file(source.cmake, dir=output, muna=muna)
        _write_file(source.readme, dir=output, muna=muna)
        _write_file(source.example, dir=output, muna=muna)
        if source.resources:
            resource_path = output / "resources"
            resource_path.mkdir()
            for res in source.resources:
                _write_file(res.url, name=res.name, dir=resource_path, muna=muna)

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

def _compute_predictor_url(api_url: str, tag: str) -> str:
    parsed_url = urlparse(api_url)
    hostname_parts = parsed_url.hostname.split(".")
    if hostname_parts[0] == "api":
        hostname_parts.pop(0)
    hostname = ".".join(hostname_parts)
    netloc = hostname if not parsed_url.port else f"{hostname}:{parsed_url.port}"
    predictor_url = urlunparse(parsed_url._replace(netloc=netloc, path=f"{tag}"))
    return predictor_url

def _get_current_target() -> str:
    match (platform.system().lower(), platform.machine().lower()):
        case ("darwin", "arm64"):       return "arm64-apple-darwin"
        case ("linux", "aarch64"):      return "aarch64-unknown-linux-gnu"
        case ("linux", "x86_64"):       return "x86_64-unknown-linux-gnu"
        case ("windows", "arm64"):      return "aarch64-pc-windows-msvc"
        case ("windows", "amd64"):      return "x86_64-pc-windows-msvc"
        case (system, arch):            raise ValueError(f"Cannot transpile because your system target is unsupported: {system} {arch}")

def _write_file(
    url: str,
    *,
    name: str=None,
    dir: Path,
    muna: Muna,
) -> Path:
    name = name or Path(urlparse(url).path).name
    path = dir / name
    muna.client.download(url, path, progress=True)
    return path

def _resolve_source_path(
    path: str,
    *,
    trust_remote_code: bool
) -> Path:
    # GitHub URL
    github_match = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", path)
    if github_match:
        if not trust_remote_code:
            raise ValueError(
                "Cannot transpile remote code without explicit trust. "
                "Pass --trust-remote-code to confirm you trust this code."
            )
        owner, repo, ref, file_path = github_match.groups()
        return _download_github_file(owner, repo, ref, file_path)
    # Local path
    local_path = Path(path).resolve()
    if not local_path.exists():
        raise ValueError(f"Cannot transpile because no file exists at path: {local_path}")
    if not local_path.is_file():
        raise ValueError(f"Cannot transpile because path is not a file: {local_path}")
    return local_path

def _download_github_file(
    owner: str,
    repo: str,
    ref: str,
    file_path: str
) -> Path:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = { "Accept": "application/vnd.github.v3+json" }
    github_token = environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    response = http_get(api_url, headers=headers, params={ "ref": ref })
    if response.status_code == 404:
        raise ValueError(
            f"Cannot transpile because file not found: {file_path} at ref '{ref}' "
            f"in repository {owner}/{repo}. Note: branch names containing '/' are not supported."
        )
    response.raise_for_status()
    data = response.json()
    # Decode content
    if data.get("encoding") == "base64":
        content = b64decode(data["content"])
    else:
        download_response = http_get(data["download_url"], headers=headers)
        download_response.raise_for_status()
        content = download_response.content
    # Write to temp file
    with TemporaryDirectory(delete=False) as tmp_dir:
        name = Path(file_path).name
        path = Path(tmp_dir) / name
        path.write_bytes(content)
        return path

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

class _TranspiledSource(BaseModel):
    code: str
    cmake: str
    readme: str
    example: str
    resources: list[PredictionResource]

class _SourceEvent(BaseModel):
    event: Literal["sources"]
    data: list[_TranspiledSource]

class CompileError(Exception):
    pass

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
        task = CustomProgressTask(loading_text=event.data.message)
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