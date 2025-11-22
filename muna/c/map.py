#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ctypes import byref, c_int32, c_void_p, create_string_buffer
from typing import final

from .fxnc import get_fxnc, status_to_error, FXNStatus
from .value import Value

@final
class ValueMap:

    def __init__(self, map=None, *, owner: bool=True):
        if map is None:
            map = c_void_p()
            owner = True
            status = get_fxnc().FXNValueMapCreate(byref(map))
            if status != FXNStatus.OK:
                raise RuntimeError(f"Failed to create value map with error: {status_to_error(status)}")
        self.__map = map
        self.__owner = owner

    def key(self, index: int) -> str:
        buffer = create_string_buffer(256)
        status = get_fxnc().FXNValueMapGetKey(self.__map, index, buffer, len(buffer))
        if status == FXNStatus.OK:
            return buffer.value.decode("utf-8")
        else:
            raise RuntimeError(f"Failed to get value map key at index {index} with error: {status_to_error(status)}")

    def __getitem__(self, key: str) -> Value | None:
        value = c_void_p()
        status = get_fxnc().FXNValueMapGetValue(self.__map, key.encode(), byref(value))
        if status == FXNStatus.OK:
            return Value(value, owner=False)
        else:
            raise RuntimeError(f"Failed to get value map value for key '{key}' with error: {status_to_error(status)}")

    def __setitem__(self, key: str, value: Value):
        status = get_fxnc().FXNValueMapSetValue(self.__map, key.encode(), value._Value__value)
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to set value map value for key '{key}' with error: {status_to_error(status)}")

    def __len__(self) -> int:
        count = c_int32()
        status = get_fxnc().FXNValueMapGetSize(self.__map, byref(count))
        if status == FXNStatus.OK:
            return count.value
        else:
            raise RuntimeError(f"Failed to get value map size with error: {status_to_error(status)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__release()

    def __release(self):
        if self.__map and self.__owner:
            get_fxnc().FXNValueMapRelease(self.__map)
        self.__map = None