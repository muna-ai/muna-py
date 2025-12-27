#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from enum import IntFlag
from ctypes import byref, cast, c_char_p, c_int, c_int32, c_uint8, c_void_p, string_at, POINTER
from io import BytesIO
from json import dumps, loads
from numpy import array, dtype, generic, int32, ndarray, zeros
from numpy.ctypeslib import as_array, as_ctypes_type
from PIL import Image
from pydantic import BaseModel
from typing import final

from ..types import Dtype, Value as Object
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
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get value data with error: {status_to_error(status)}")
        return data            

    @property
    def type(self) -> Dtype:
        dtype = c_int()
        status = get_fxnc().FXNValueGetType(self.__value, byref(dtype))
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get value data type with error: {status_to_error(status)}")
        return _dtype_from_c(dtype.value)        

    @property
    def shape(self) -> tuple[int, ...] | None:
        if self.type not in _TENSOR_ISH_DTYPES:
            return None
        fxnc = get_fxnc()
        dims = c_int32()
        status = fxnc.FXNValueGetDimensions(self.__value, byref(dims))
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get value dimensions with error: {status_to_error(status)}")
        shape = zeros(dims.value, dtype=int32)
        status = fxnc.FXNValueGetShape(self.__value, shape.ctypes.data_as(POINTER(c_int32)), dims)
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to get value shape with error: {status_to_error(status)}")
        return tuple(shape)

    def to_object(self) -> Object:
        match self.type:
            case Dtype.null:    return None
            case t if t in _TENSOR_DTYPES:
                ctype = as_ctypes_type(dtype(type))
                tensor = as_array(cast(self.data, POINTER(ctype)), self.shape)
                return tensor.copy() if len(tensor.shape) else tensor.item()
            case Dtype.string:  return string_at(self.data).decode()
            case Dtype.list:    return loads(string_at(self.data))
            case Dtype.dict:    return loads(string_at(self.data))
            case Dtype.image:
                data = as_array(cast(self.data, POINTER(c_uint8)), self.shape).copy()
                return Image.fromarray(data.squeeze())
            case Dtype.binary:  return string_at(self.data, self.shape[0])
            case _:             raise ValueError(f"Failed to convert value with type `{self.type}` to object because it is not supported")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__release()

    def __release(self):
        if self.__value and self.__owner:
            get_fxnc().FXNValueRelease(self.__value)
        self.__value = None

    @classmethod
    def from_object(
        cls,
        obj: Object,
        *,
        flags: ValueFlags=ValueFlags.NONE
    ) -> Value:
        value = c_void_p()
        obj = _ensure_object_serializable(obj)
        match obj:
            case None:      status = get_fxnc().FXNValueCreateNull(byref(value))
            case float():   return cls.from_object(array(obj, dtype=Dtype.float32), flags=flags | ValueFlags.COPY_DATA)
            case bool():    return cls.from_object(array(obj, dtype=Dtype.bool), flags=flags | ValueFlags.COPY_DATA)
            case int():     return cls.from_object(array(obj, dtype=Dtype.int32), flags=flags | ValueFlags.COPY_DATA)
            case generic(): return cls.from_object(array(obj), flags=flags | ValueFlags.COPY_DATA)            
            case str():     status = get_fxnc().FXNValueCreateString(obj.encode(), byref(value))
            case list():    status = get_fxnc().FXNValueCreateList(dumps(obj).encode(), byref(value))
            case dict():    status = get_fxnc().FXNValueCreateDict(dumps(obj).encode(), byref(value))
            case BytesIO(): return cls.from_object(obj.getvalue(), flags=flags | ValueFlags.COPY_DATA)
            case bytes():   status = get_fxnc().FXNValueCreateBinary(c_char_p(obj), len(obj), flags, byref(value))
            case ndarray():
                status = get_fxnc().FXNValueCreateArray(
                    obj.ctypes.data_as(c_void_p),
                    obj.ctypes.shape_as(c_int32),
                    len(obj.shape),
                    _dtype_to_c(obj.dtype.name),
                    flags,
                    byref(value)
                )
            case Image.Image():
                tensor = array(obj)
                status = get_fxnc().FXNValueCreateImage(
                    tensor.ctypes.data_as(c_void_p),
                    obj.width,
                    obj.height,
                    tensor.shape[2],
                    flags,
                    byref(value)
                )
            case _:
                raise ValueFlags(f"Failed to convert object to prediction value because object has an unsupported type: {type(obj)}")
        if status != FXNStatus.OK:
            raise RuntimeError(f"Failed to create string value with error: {status_to_error(status)}")
        return Value(value)

def _ensure_object_serializable(obj: object) -> object:
    is_dict = is_dataclass(obj) and not isinstance(obj, type)
    match obj:
        case list():        return list(map(_ensure_object_serializable, obj))
        case BaseModel():   return obj.model_dump(mode="json", by_alias=True)
        case _ if is_dict:  return asdict(obj)
        case _:             return obj

def _dtype_to_c(type: Dtype) -> int:
    match type:
        case Dtype.null:        return 0
        case Dtype.float16:     return 1
        case Dtype.float32:     return 2
        case Dtype.float64:     return 3
        case Dtype.int8:        return 4
        case Dtype.int16:       return 5
        case Dtype.int32:       return 6
        case Dtype.int64:       return 7
        case Dtype.uint8:       return 8
        case Dtype.uint16:      return 9
        case Dtype.uint32:      return 10
        case Dtype.uint64:      return 11
        case Dtype.bool:        return 12
        case Dtype.string:      return 13
        case Dtype.list:        return 14
        case Dtype.dict:        return 15
        case Dtype.image:       return 16
        case Dtype.binary:      return 17
        case Dtype.bfloat16:    return 18
        case _:                 raise ValueError(f"Failed to convert data type because it is not supported: {type}")

def _dtype_from_c(type: int) -> Dtype:
    match type:
        case 0:     return Dtype.null
        case 1:     return Dtype.float16
        case 2:     return Dtype.float32
        case 3:     return Dtype.float64
        case 4:     return Dtype.int8
        case 5:     return Dtype.int16
        case 6:     return Dtype.int32
        case 7:     return Dtype.int64
        case 8:     return Dtype.uint8
        case 9:     return Dtype.uint16
        case 10:    return Dtype.uint32
        case 11:    return Dtype.uint64
        case 12:    return Dtype.bool
        case 13:    return Dtype.string
        case 14:    return Dtype.list
        case 15:    return Dtype.dict
        case 16:    return Dtype.image
        case 17:    return Dtype.binary
        case 18:    return Dtype.bfloat16
        case _:     raise ValueError(f"Failed to convert data type because it is not supported: {type}")

_TENSOR_DTYPES = {
    Dtype.bfloat16, Dtype.float16, Dtype.float32, Dtype.float64,
    Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
    Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64,
    Dtype.bool,
}
_TENSOR_ISH_DTYPES = _TENSOR_DTYPES | { Dtype.image, Dtype.binary }