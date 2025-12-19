# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from base64 import b64encode
from ctypes import c_uint8, cast, string_at, POINTER
from dataclasses import asdict, is_dataclass
from io import BytesIO
from json import dumps, loads
from numpy import array, dtype, generic, load as npz_load, ndarray, savez_compressed
from numpy.ctypeslib import as_array, as_ctypes_type
from numpy.lib.npyio import NpzFile
from PIL import Image
from pydantic import BaseModel
from requests import get
from urllib.request import urlopen

from ..c import Value as LocalValue, ValueFlags
from ..types import Dtype, RemoteValue, Value

def create_local_value(
    obj: Value,
    *,
    flags: ValueFlags=ValueFlags.NONE
) -> LocalValue:
    """
    Create a local value from an object.
    """
    obj = _ensure_object_serializable(obj)
    match obj:
        case None:          return LocalValue.create_null()
        case float():       return create_local_value(array(obj, dtype=Dtype.float32), flags=flags | ValueFlags.COPY_DATA)
        case bool():        return create_local_value(array(obj, dtype=Dtype.bool), flags=flags | ValueFlags.COPY_DATA)
        case int():         return create_local_value(array(obj, dtype=Dtype.int32), flags=flags | ValueFlags.COPY_DATA)
        case generic():     return create_local_value(array(obj), flags=flags | ValueFlags.COPY_DATA)
        case ndarray():     return LocalValue.create_array(obj, flags=flags)
        case str():         return LocalValue.create_string(obj)
        case list():        return LocalValue.create_list(obj)
        case dict():        return LocalValue.create_dict(obj)
        case Image.Image(): return LocalValue.create_image(obj)
        case memoryview():  return LocalValue.create_binary(obj, flags=flags)
        case bytes():       return LocalValue.create_binary(memoryview(obj), flags=flags | ValueFlags.COPY_DATA)
        case bytearray():   return LocalValue.create_binary(memoryview(obj), flags=flags | ValueFlags.COPY_DATA)
        case BytesIO():     return LocalValue.create_binary(memoryview(obj.getvalue()), flags=flags | ValueFlags.COPY_DATA)
        case _:             raise RuntimeError(f"Failed to convert object to prediction value because object has an unsupported type: {type(obj)}")

def create_remote_value(obj: Value) -> RemoteValue:
    """
    Create a remote value from an object.
    """
    obj = _ensure_object_serializable(obj)
    match obj:
        case None:      return RemoteValue(data=None, type=Dtype.null)
        case float():   return create_remote_value(array(obj, dtype=Dtype.float32))
        case bool():    return create_remote_value(array(obj, dtype=Dtype.bool))
        case int():     return create_remote_value(array(obj, dtype=Dtype.int32))
        case generic(): return create_remote_value(array(obj))
        case ndarray():
            buffer = BytesIO()
            savez_compressed(buffer, obj, allow_pickle=False)
            data = _upload_value_data(buffer)
            return RemoteValue(data=data, type=obj.dtype.name)
        case str():
            buffer = BytesIO(obj.encode())
            data = _upload_value_data(buffer, mime="text/plain")
            return RemoteValue(data=data, type=Dtype.string)
        case list():
            buffer = BytesIO(dumps(obj).encode())
            data = _upload_value_data(buffer, mime="application/json")
            return RemoteValue(data=data, type=Dtype.list)
        case dict():
            buffer = BytesIO(dumps(obj).encode())
            data = _upload_value_data(buffer, mime="application/json")
            return RemoteValue(data=data, type=Dtype.dict)
        case Image.Image():
            buffer = BytesIO()
            format = "PNG" if obj.mode == "RGBA" else "JPEG"
            mime = f"image/{format.lower()}"
            obj.save(buffer, format=format)
            data = _upload_value_data(buffer, mime=mime)
            return RemoteValue(data=data, type=Dtype.image)
        case BytesIO():
            data = _upload_value_data(obj)
            return RemoteValue(data=data, type=Dtype.binary)
        case _:
            raise ValueError(f"Failed to serialize value '{obj}' of type `{type(obj)}` because it is not supported")

def parse_local_value(value: LocalValue) -> Value:
    """
    Parse an object from a local value.
    """
    is_tensor = value.type in _TENSOR_DTYPES
    is_json = value.type in _JSON_TYPES
    match value.type:
        case Dtype.null:    return None
        case _ if is_tensor:
            ctype = as_ctypes_type(dtype(type))
            tensor = as_array(cast(value.data, POINTER(ctype)), value.shape)
            return tensor.copy() if len(tensor.shape) else tensor.item()
        case Dtype.string:  return string_at(value.data).decode()
        case _ if is_json:  return loads(string_at(value.data))
        case Dtype.image:
            data = as_array(cast(value.data, POINTER(c_uint8)), value.shape).copy()
            return Image.fromarray(data.squeeze())
        case Dtype.binary:  return BytesIO(string_at(value.data, value.shape[0]))
        case _:             raise ValueError(f"Failed to parse local value with type `{value.type}` because it is not supported")

def parse_remote_value(value: RemoteValue) -> Value:
    """
    Parse an object from a remote value.
    """
    buffer = _download_value_data(value.data) if value.data else None
    is_tensor = value.type in _TENSOR_DTYPES
    is_json = value.type in _JSON_TYPES
    match value.type:
        case Dtype.null:    return None
        case _ if is_tensor:
            archive: NpzFile = npz_load(buffer)
            array = next(iter(archive.values()))
            return array if len(array.shape) else array.item()
        case Dtype.string:  return buffer.getvalue().decode("utf-8")
        case _ if is_json:  return loads(buffer.getvalue().decode("utf-8"))
        case Dtype.image:   return Image.open(buffer)
        case Dtype.binary:  return buffer
        case _:             raise ValueError(f"Failed to parse remote value with type `{value.type}` because it is not supported")

def _ensure_object_serializable(obj: object) -> object:
    """
    Ensure an object is serializable.
    """
    is_dict = is_dataclass(obj) and not isinstance(obj, type)
    match obj:
        case list():        return list(map(_ensure_object_serializable, obj))
        case BaseModel():   return obj.model_dump(mode="json", by_alias=True)
        case _ if is_dict:  return asdict(obj)
        case _:             return obj

def _upload_value_data(
    data: BytesIO,
    *,
    mime: str="application/octet-stream"
) -> str:
    """
    Upload value data to a URL.
    """
    encoded_data = b64encode(data.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded_data}"

def _download_value_data(url: str) -> BytesIO:
    """
    Download value data from a URL.
    """
    if url.startswith("data:"):
        with urlopen(url) as response:
            return BytesIO(response.read())
    response = get(url)
    response.raise_for_status()
    result = BytesIO(response.content)
    return result

_TENSOR_DTYPES = {
    Dtype.bfloat16, Dtype.float16, Dtype.float32, Dtype.float64,
    Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
    Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64,
    Dtype.bool,
}
_JSON_TYPES = { Dtype.list, Dtype.dict }