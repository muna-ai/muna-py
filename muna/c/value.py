#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Iterable
from enum import IntFlag
from ctypes import byref, cast, c_char_p, c_int, c_int32, c_uint8, c_void_p, string_at, POINTER
from io import BytesIO
from json import dumps, loads
from numpy import array, dtype, int32, ndarray, zeros
from numpy.ctypeslib import as_array, as_ctypes_type
from PIL import Image
from typing import final, Any

from ..types import Dtype
from .fxnc import get_fxnc, status_to_error, FXNStatus

class ValueFlags(IntFlag):
    NONE = 0
    COPY_DATA = 1

@final
class Value:

    def __init__(self, value, *, owner: bool=True):
        self.__value = value
        self.__owner = owner

    @property
    def data(self):
        data = c_void_p()
        status = get_fxnc().FXNValueGetData(self.__value, byref(data))
        if status == FXNStatus.OK:
            return data
        else:
            raise RuntimeError(f"Failed to get value data with error: {status_to_error(status)}")

    @property
    def type(self) -> Dtype:
        dtype = c_int()
        status = get_fxnc().FXNValueGetType(self.__value, byref(dtype))
        if status == FXNStatus.OK:
            return _DTYPE_TO_STR.get(dtype.value)
        else:
            raise RuntimeError(f"Failed to get value data type with error: {status_to_error(status)}")

    @property
    def shape(self) -> list[int] | None:
        if self.type not in _TENSOR_ISH_DTYPES:
            return None
        fxnc = get_fxnc()
        dims = c_int32()
        status = fxnc.FXNValueGetDimensions(self.__value, byref(dims))
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get value dimensions with error: {status_to_error(status)}")
        shape = zeros(dims.value, dtype=int32)
        status = fxnc.FXNValueGetShape(self.__value, shape.ctypes.data_as(POINTER(c_int32)), dims)
        if status == FXNStatus.OK:
            return shape.tolist()
        else:
            raise RuntimeError(f"Failed to get value shape with error: {status_to_error(status)}")

    def to_object(self) -> Any:
        type = self.type
        if type == Dtype.null:
            return None
        elif type in _TENSOR_DTYPES:
            ctype = as_ctypes_type(dtype(type))
            tensor = as_array(cast(self.data, POINTER(ctype)), self.shape)
            return tensor.item() if len(tensor.shape) == 0 else tensor.copy()
        elif type == Dtype.string:
            return cast(self.data, c_char_p).value.decode()
        elif type in [Dtype.list, Dtype.dict]:
            return loads(cast(self.data, c_char_p).value.decode())
        elif type == Dtype.image:
            pixel_buffer = as_array(cast(self.data, POINTER(c_uint8)), self.shape)
            return Image.fromarray(pixel_buffer.squeeze()).copy()
        elif type == Dtype.binary:
            return BytesIO(string_at(self.data, self.shape[0]))
        else:
            raise RuntimeError(f"Failed to convert Function value to object because value has unsupported type: {type}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__release()

    def __release(self):
        if self.__value and self.__owner:
            get_fxnc().FXNValueRelease(self.__value)
        self.__value = None

    @classmethod
    def create_array(
        cls,
        data: ndarray,
        *,
        flags: ValueFlags=ValueFlags.NONE
    ) -> Value:
        dtype = _STR_TO_DTYPE.get(data.dtype.name)
        if dtype is None:
            raise RuntimeError(f"Failed to create array value because data type is not supported: {data.dtype}")
        value = c_void_p()
        status = get_fxnc().FXNValueCreateArray(
            data.ctypes.data_as(c_void_p),
            data.ctypes.shape_as(c_int32),
            len(data.shape),
            dtype,
            flags,
            byref(value)
        )
        if status == FXNStatus.OK:
            return Value(value)
        else:
            raise RuntimeError(f"Failed to create array value with error: {status_to_error(status)}")

    @classmethod
    def create_string(cls, data: str) -> Value:
        value = c_void_p()
        status = get_fxnc().FXNValueCreateString(data.encode(), byref(value))
        if status == FXNStatus.OK:
            return Value(value)
        else:
            raise RuntimeError(f"Failed to create string value with error: {status_to_error(status)}")

    @classmethod
    def create_list(cls, data: Iterable[Any]) -> Value:
        value = c_void_p()
        status = get_fxnc().FXNValueCreateList(dumps(data).encode(), byref(value))
        if status == FXNStatus.OK:
            return Value(value)
        else:
            raise RuntimeError(f"Failed to create list value with error: {status_to_error(status)}")
    
    @classmethod
    def create_dict(cls, data: dict[str, Any]) -> Value:
        value = c_void_p()
        status = get_fxnc().FXNValueCreateDict(dumps(data).encode(), byref(value))
        if status == FXNStatus.OK:
            return Value(value)
        else:
            raise RuntimeError(f"Failed to create dict value with error: {status_to_error(status)}")

    @classmethod
    def create_image(cls, image: Image.Image) -> Value:
        value = c_void_p()
        pixel_buffer = array(image)
        status = get_fxnc().FXNValueCreateImage(
            pixel_buffer.ctypes.data_as(c_void_p),
            image.width,
            image.height,
            pixel_buffer.shape[2],
            ValueFlags.COPY_DATA,
            byref(value)
        )
        if status == FXNStatus.OK:
            return Value(value)
        else:
            raise RuntimeError(f"Failed to create image value with error: {status_to_error(status)}")        

    @classmethod
    def create_binary(
        cls,
        data: memoryview,
        *,
        flags: ValueFlags=ValueFlags.NONE
    ) -> Value:
        buffer = (c_uint8 * len(data)).from_buffer(data)
        value = c_void_p()
        status = get_fxnc().FXNValueCreateBinary(buffer, len(data), flags, byref(value))
        if status == FXNStatus.OK:
            return Value(value)
        else:
            raise RuntimeError(f"Failed to create binary value with error: {status_to_error(status)}")

    @classmethod
    def create_null(cls) -> Value:
        value = c_void_p()
        status = get_fxnc().FXNValueCreateNull(byref(value))
        if status == FXNStatus.OK:
            return Value(value)
        else:
            raise RuntimeError(f"Failed to create null value with error: {status_to_error(status)}")


_STR_TO_DTYPE = {
    Dtype.null: 0,
    Dtype.float16: 1,
    Dtype.float32: 2,
    Dtype.float64: 3,
    Dtype.int8: 4,
    Dtype.int16: 5,
    Dtype.int32: 6,
    Dtype.int64: 7,
    Dtype.uint8: 8,
    Dtype.uint16: 9,
    Dtype.uint32: 10,
    Dtype.uint64: 11,
    Dtype.bool: 12,
    Dtype.string: 13,
    Dtype.list: 14,
    Dtype.dict: 15,
    Dtype.image: 16,
    Dtype.binary: 17,
}
_DTYPE_TO_STR = { value: key for key, value in _STR_TO_DTYPE.items() }
_TENSOR_DTYPES = {
    Dtype.float16,
    Dtype.float32,
    Dtype.float64,
    Dtype.int8,
    Dtype.int16,
    Dtype.int32,
    Dtype.int64,
    Dtype.uint8,
    Dtype.uint16,
    Dtype.uint32,
    Dtype.uint64,
    Dtype.bool,
}
_TENSOR_ISH_DTYPES = _TENSOR_DTYPES | { Dtype.image }