#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ctypes import CDLL
from enum import IntEnum
from importlib import resources
from platform import machine, system

_fxnc: CDLL = None

class FXNStatus(IntEnum):
    OK = 0
    ERROR_INVALID_ARGUMENT = 1
    ERROR_INVALID_OPERATION = 2
    ERROR_NOT_IMPLEMENTED = 3

def get_fxnc() -> CDLL:
    global _fxnc
    _fxnc = _fxnc if _fxnc is not None else _load_fxnc()
    return _fxnc

def set_fxnc(fxnc: CDLL):
    global _fxnc
    _fxnc = fxnc

def _load_fxnc() -> CDLL:
    os = system().lower()
    os = "macos" if os == "darwin" else os
    arch = machine().lower()
    arch = "arm64" if arch == "aarch64" else arch
    arch = "x86_64" if arch in ["x64", "amd64"] else arch
    package = f"muna.lib.{os}.{arch}"
    resource = "libFunction.so"
    resource = "Function.dylib" if os == "macos" else resource
    resource = "Function.dll" if os == "windows" else resource
    with resources.path(package, resource) as path:
        return CDLL(str(path))
    
def status_to_error(status: int) -> str:
    if status == FXNStatus.ERROR_INVALID_ARGUMENT:
        return "FXN_ERROR_INVALID_ARGUMENT"
    elif status == FXNStatus.ERROR_INVALID_OPERATION:
        return "FXN_ERROR_INVALID_OPERATION"
    elif status == FXNStatus.ERROR_NOT_IMPLEMENTED:
        return "FXN_ERROR_NOT_IMPLEMENTED"
    return ""