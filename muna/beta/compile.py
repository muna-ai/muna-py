# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable
from contextvars import ContextVar
from enum import IntFlag
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr
from rich.progress import (
    BarColumn, DownloadColumn, TimeRemainingColumn,
    TransferSpeedColumn
)
import shutil
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import cast, Literal, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

from ..logging import CustomProgressTask, run_streamed

_active_registry: ContextVar[list | None] = ContextVar(
    "muna_active_compile_registry",
    default=None,
)
_baseline_registry: list = []

class Platform(IntFlag):
    """
    Compile target.
    """
    UNKNOWN         = 0
    ANDROID_ARM     = 1 << 0
    ANDROID_ARM64   = 1 << 1
    ANDROID_X86     = 1 << 2 # deprecated
    ANDROID_X64     = 1 << 3 # deprecated
    ANDROID         = ANDROID_ARM | ANDROID_ARM64
    IOS_ARM64       = 1 << 4
    IOS             = IOS_ARM64
    MACOS_X64       = 1 << 5 # deprecated
    MACOS_ARM64     = 1 << 6
    MACOS           = MACOS_ARM64
    LINUX_X64       = 1 << 7
    LINUX_ARM64     = 1 << 8
    LINUX           = LINUX_X64 | LINUX_ARM64
    VISIONOS_ARM64  = 1 << 13
    VISIONOS        = VISIONOS_ARM64
    WASM32          = 1 << 9
    WASM64          = 1 << 12
    WASM            = WASM32 | WASM64
    WINDOWS_X64     = 1 << 10
    WINDOWS_ARM64   = 1 << 11
    WINDOWS         = WINDOWS_X64 | WINDOWS_ARM64
    ALL             = ANDROID | IOS | MACOS | LINUX | VISIONOS | WASM | WINDOWS # -1 might have issues in Pydantic

CompileTarget = Literal[
    "android",
    "ios",
    "linux",
    "macos",
    "visionos",
    "wasm",
    "windows"
] | str

class CompileLibrary(BaseModel, **ConfigDict(frozen=True)):
    """
    Compile library.
    """
    target: str = Field(description="CMake target name to link to.")
    include: str | None = Field(
        default=None,
        description=dedent("""
        Path to a CMake fragment that defines `target`. 
        Resolved relative to the dialect's `cmake/` directory. 
        """).strip()
    )
    platform: Platform = Field(
        default=Platform.ALL,
        description="Platforms supported by this library."
    )
    order: int = Field(
        default=0,
        description="Library linking order. Defaults to 0."
    )
    definitions: dict[str, str] = Field(
        default_factory=dict,
        description="CMake definitions to pass when building with this library."
    )

    def __hash__(self) -> int:
        return hash((
            self.target,
            self.platform,
            self.include,
            self.order,
            frozenset(self.definitions.items()),
        ))

class CompileResource(BaseModel, **ConfigDict(frozen=True)):
    """
    Compile resource.
    """
    _uid: str = PrivateAttr(default_factory=lambda: uuid4().hex[:16])
    data: bytes | None = Field(default=None, repr=False, description="Resource data.")
    path: Path | None = Field(default=None, description="Resource path.")
    url: str | None = Field(default=None, description="Resource URL.")
    sha256: str | None = Field(default=None, description="Resource SHA256 checksum.")

    @property
    def id(self) -> str:
        """
        Unique identifier for this resource.
        """
        return f"res_{self._uid}"

    @classmethod
    def from_bytes(cls, data: bytes) -> CompileResource:
        """
        Create a compile resource from data in memory.
        """
        return cls(data=data)

    @classmethod
    def from_path(cls, path: Path) -> CompileResource:
        """
        Create a compile resource from a file at a given path.
        """
        return cls(path=path)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        sha256: str | None=None
    ) -> CompileResource:
        """
        Create a compile resource from a file at a given URL.
        """
        return cls(url=url, sha256=sha256)

    @classmethod
    def from_hf_hub(
        cls,
        repo_id: str,
        *files: str,
        revision: str | None=None
    ) -> tuple[CompileResource, ...]:
        """
        Create compile resources from a file in a Huggingface Hub repository.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise RuntimeError("The `huggingface_hub` package is required to create a compile resource from a Huggingface Hub repository. Install it with `pip install huggingface_hub`.")
        api = HfApi()
        revision = revision or api.repo_info(repo_id).sha
        sha_by_path = {
            p.path: (p.lfs.sha256 if getattr(p, "lfs", None) else None)
            for p in api.get_paths_info(repo_id, list(files), revision=revision)
        }
        resources = [cls(
            url=f"hf://{repo_id}@{revision}/{f}",
            sha256=sha_by_path.get(f),
        ) for f in files]
        return tuple(resources)

class CompileConstant(BaseModel, **ConfigDict(frozen=True)):
    """
    Compile constant.
    """
    type: str = Field(description="Compiled type of the emitted constant.")
    value: str = Field(description="Expression that instantiates the constant.")
    platform: Platform = Field(
        default=Platform.ALL,
        description="Platforms on which this constant can be instantiated."
    )
    includes: tuple[str, ...] = Field(
        default=(),
        description=dedent("""
        Required header paths.
        Resolved relative to the dialect's `src/` directory. 
        """).strip()
    )
    libraries: tuple[CompileLibrary, ...] = Field(
        default=(),
        description="Libraries required by the constant."
    )
    resources: tuple[CompileResource, ...] = Field(
        default=(),
        description="Resources required by the constant."
    )
    meta: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Constant metadata."
    )

class CompileDialect(BaseModel, **ConfigDict(frozen=True)):
    """
    Compile dialect.
    """
    kind: Literal["builtin", "url"]
    url: str | None = Field(default=None, description="Dialect URL.")
    sha256: str | None = Field(default=None, description="Dialect tarball checksum.")
    _path: Path | None = PrivateAttr(default=None)
    _git_url: str | None = PrivateAttr(default=None)
    _git_revision: str | None = PrivateAttr(default=None)
    _git_subdir: str | None = PrivateAttr(default=None)

    @classmethod
    def from_builtin(cls) -> CompileDialect:
        """
        Muna's built-in dialect.
        """
        return cls(kind="builtin")

    @classmethod
    def from_path(cls, path: Path) -> CompileDialect:
        """
        Create a dialect rooted at a local directory.
        """
        instance = cls(kind="url")
        instance._path = Path(path).resolve()
        return instance

    @classmethod
    def from_git(
        cls,
        url: str,
        revision: str | None = None,
        subdir: str | None = None
    ) -> CompileDialect:
        """
        Create a dialect rooted at a Git repository.
        """
        instance = cls(kind="url")
        instance._git_url = url
        instance._git_revision = revision
        instance._git_subdir = subdir
        return instance

    @classmethod
    def from_url(
        cls,
        url: str,
        sha256: str | None = None
    ) -> CompileDialect:
        """
        Create a dialect rooted at a zip or tarball URL.
        """
        return cls(kind="url", url=url, sha256=sha256)

    def populate(self, muna=None) -> CompileDialect:
        """
        Upload this dialect.
        """
        from ..muna import Muna
        if self.kind == "builtin":
            return self
        if self.url is not None:
            return self
        muna = cast(Muna, muna) or Muna()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            if self._path is not None:
                source_root = self._path
            elif self._git_url is not None:
                clone_dir = tmp_path / "clone"
                run_streamed([
                    "git", "clone", "--depth", "1", #"--progress",
                    *(["--branch", self._git_revision] if self._git_revision else []),
                    self._git_url, str(clone_dir),
                ])
                source_root = clone_dir / self._git_subdir if self._git_subdir else clone_dir
            else:
                raise ValueError("CompileDialect has no source set")
            archive_path = shutil.make_archive(
                base_name=str(tmp_path / "dialect"),
                format="gztar",
                root_dir=str(source_root),
            )
            file_size = Path(archive_path).stat().st_size
            with CustomProgressTask(
                loading_text=f"[grey50]Uploading {Path(archive_path).name}[/grey50]",
                columns=[
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                ],
            ) as upload_task:
                upload_task.update(total=file_size)
                archive_url = muna.client.upload(
                    archive_path,
                    progress=False,
                    on_progress=lambda n: upload_task.update(advance=n),
                )
            archive_checksum = next(
                component
                for component in reversed(urlparse(archive_url).path.split('/'))
                if component
            )
        return CompileDialect(
            kind="url",
            url=archive_url,
            sha256=archive_checksum
        )

T = TypeVar("T")

def compile_constant(
    object_type: type[T] | Callable[[], type[T] | None],
    *,
    strict: bool = False
) -> Callable[
    [Callable[[T], CompileConstant | None]],
    Callable[[T], CompileConstant | None],
]:
    """
    Register a factory that lowers Python values of type `T` to a `CompileConstant`.
    
    The decorated function may return `None` to indicate it doesn't 
    apply to the given value (e.g. filtering by identity).

    Parameters:
        object_type (type | Callable): Type whose instances this factory handles.
        strict (bool): Whether to only match the exact type and not subclasses.
    """
    def decorator(
        fn: Callable[[T], CompileConstant | None],
    ) -> Callable[[T], CompileConstant | None]:
        _get_registry().append((object_type, fn, strict))
        return fn
    return decorator

def _get_registry() -> list:
    active = _active_registry.get()
    return _baseline_registry if active is None else active