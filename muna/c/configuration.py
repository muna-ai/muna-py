#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ctypes import byref, c_int, c_void_p, create_string_buffer
from pathlib import Path
from typing import final

from ..types import Acceleration
from .fxnc import get_fxnc, status_to_error, FXNStatus

@final
class Configuration:

    def __init__(self):
        configuration = c_void_p()
        status = get_fxnc().FXNConfigurationCreate(byref(configuration))
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to create configuration with error: {status_to_error(status)}")
        self.__configuration = configuration            

    @property
    def tag(self) -> str:
        buffer = create_string_buffer(2048)
        status = get_fxnc().FXNConfigurationGetTag(
            self.__configuration,
            buffer,
            len(buffer)
        )
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get configuration tag with error: {status_to_error(status)}")
        tag = buffer.value.decode("utf-8")
        return tag if tag else None

    @tag.setter
    def tag(self, tag: str):
        tag = tag.encode() if tag is not None else None
        status = get_fxnc().FXNConfigurationSetTag(self.__configuration, tag)
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to set configuration tag with error: {status_to_error(status)}")

    @property
    def token(self) -> str:
        buffer = create_string_buffer(2048)
        status = get_fxnc().FXNConfigurationGetToken(
            self.__configuration,
            buffer,
            len(buffer)
        )
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get configuration token with error: {status_to_error(status)}")
        token = buffer.value.decode("utf-8")
        return token if token else None

    @token.setter
    def token(self, token: str):
        token = token.encode() if token is not None else None
        status = get_fxnc().FXNConfigurationSetToken(self.__configuration, token)
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to set configuration token with error: {status_to_error(status)}")            

    @property
    def acceleration(self) -> Acceleration:
        acceleration = c_int()
        status = get_fxnc().FXNConfigurationGetAcceleration(
            self.__configuration,
            byref(acceleration)
        )
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get configuration acceleration with error: {status_to_error(status)}")
        return self.__to_acceleration_str(acceleration.value)

    @acceleration.setter
    def acceleration(self, acceleration: Acceleration):
        status = get_fxnc().FXNConfigurationSetAcceleration(
            self.__configuration,
            self.__to_acceleration_int(acceleration)
        )
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to set configuration acceleration with error: {status_to_error(status)}")

    @property
    def device(self):
        device = c_void_p()
        status = get_fxnc().FXNConfigurationGetDevice(
            self.__configuration,
            byref(device)
        )
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get configuration device with error: {status_to_error(status)}")
        return device if device.value else None            

    @device.setter
    def device(self, device):
        status = get_fxnc().FXNConfigurationSetDevice(self.__configuration, device)
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to set configuration device with error: {status_to_error(status)}")

    def add_resource(self, type: str, path: Path):
        status = get_fxnc().FXNConfigurationAddResource(
            self.__configuration,
            type.encode(),
            str(path).encode()
        )
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to add configuration resource with error: {status_to_error(status)}")

    @classmethod
    def get_unique_id(cls) -> str:
        buffer = create_string_buffer(2048)
        status = get_fxnc().FXNConfigurationGetUniqueID(buffer, len(buffer))
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to retrieve configuration identifier with error: {status_to_error(status)}")
        return buffer.value.decode("utf-8")

    @classmethod
    def get_client_id(cls) -> str:
        buffer = create_string_buffer(64)
        status = get_fxnc().FXNConfigurationGetClientID(buffer, len(buffer))
        if status == FXNStatus.OK:
            return buffer.value.decode("utf-8")
        else:
            raise RuntimeError(f"Failed to retrieve client identifier with error: {status_to_error(status)}")
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__release()

    def __release(self):
        if self.__configuration:
            get_fxnc().FXNConfigurationRelease(self.__configuration)
        self.__configuration = None

    def __to_acceleration_int(self, value: Acceleration) -> int:
        match value:
            case "local_auto" | "auto": return 0
            case "local_cpu" | "cpu":   return 1
            case "local_gpu" | "gpu":   return 2
            case "local_npu" | "npu":   return 4

    def __to_acceleration_str(self, value: int) -> Acceleration:
        match value:
            case 0: return "local_auto"
            case 1: return "local_cpu"
            case 2: return "local_gpu"
            case 4: return "local_npu"
            case _: return None