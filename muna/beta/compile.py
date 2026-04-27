# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable
from contextvars import ContextVar
from enum import IntFlag
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from textwrap import dedent
from typing import TypeVar
from uuid import uuid4

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
    _data: bytes | None = PrivateAttr(default=None)
    _path: Path | None = PrivateAttr(default=None)
    _url: str | None = PrivateAttr(default=None)
    _sha256: str | None = PrivateAttr(default=None)

    @property
    def id(self) -> str:
        """
        Unique identifier for this resource.
        """
        return f"MUNA_RESOURCE_{self._uid}"

    @classmethod
    def from_bytes(cls, data: bytes) -> CompileResource:
        """
        Create a compile resource from data in memory.
        """
        instance = cls()
        instance._data = data
        return instance

    @classmethod
    def from_path(cls, path: Path) -> CompileResource:
        """
        Create a compile resource from a file at a given path.
        """
        instance = cls()
        instance._path = path
        return instance

    @classmethod
    def from_url(
        cls,
        url: str,
        sha256: str=None
    ) -> CompileResource:
        """
        Create a compile resource from a file at a given URL.
        """
        instance = cls()
        instance._url = url
        instance._sha256 = sha256
        return instance

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

class CompileDialect(BaseModel, **ConfigDict(frozen=True)):
    """
    Compile dialect.
    """
    _path: Path | None = PrivateAttr(default=None)
    _git_url: str | None = PrivateAttr(default=None)
    _git_revision: str | None = PrivateAttr(default=None)
    _git_subdir: str | None = PrivateAttr(default=None)
    _url: str | None = PrivateAttr(default=None)
    _sha256: str | None = PrivateAttr(default=None)
    _builtin: bool = PrivateAttr(default=False)

    @classmethod
    def from_builtin(cls) -> CompileDialect:
        """
        Muna's built-in dialect.
        """
        instance = cls()
        instance._builtin = True
        return instance

    @classmethod
    def from_path(cls, path: Path) -> CompileDialect:
        """
        Create a dialect rooted at a local directory.
        """
        instance = cls()
        instance._path = path
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
        instance = cls()
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
        instance = cls()
        instance._url = url
        instance._sha256 = sha256
        return instance

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