#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from ctypes import addressof, c_int32, c_void_p
from typing import final

@final
class Device:
    """
    Compute device used for local prediction acceleration.

    The device is an opaque, platform-dependent pointer.
    See https://docs.muna.ai/sdk/predictions for its type on each platform.
    """

    def __init__(self, pointer: int | c_void_p):
        address = pointer.value if isinstance(pointer, c_void_p) else pointer
        if not address:
            raise ValueError("Failed to create device because pointer is null")
        self.__pointer = address
        self.__buffer: c_int32 | None = None

    @property
    def pointer(self) -> int:
        """
        Device pointer.
        """
        return self.__pointer

    @classmethod
    def cuda(cls, index: int) -> Device:
        """
        Create a CUDA or HIP device from its ordinal.
        """
        if index < 0:
            raise ValueError(f"Failed to create CUDA device because index {index} is negative")
        buffer = c_int32(index)
        device = cls(addressof(buffer))
        device.__buffer = buffer # keep alive
        return device

    @classmethod
    def parse(cls, device: Device | str) -> Device:
        """
        Parse a device from a `cuda:<index>` string.
        """
        if isinstance(device, Device):
            return device
        kind, _, index = device.partition(":")
        if kind == "cuda" and index.isdigit():
            return cls.cuda(int(index))
        raise ValueError(f"Failed to parse device `{device}`; expected `cuda:<index>`")

    def __repr__(self) -> str:
        return f"Device(pointer={self.__pointer:#x})"
