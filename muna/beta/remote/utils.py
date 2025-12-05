#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from io import BytesIO
from json import loads
from numpy import load as npz_load
from numpy.lib.npyio import NpzFile
from PIL import Image
from requests import get
from urllib.request import urlopen

from ...types import Dtype, Value
from .schema import RemoteValue

def remote_value_to_object(value: RemoteValue) -> Value:
    """
    Deserialize a remote value to an object.
    """
    if value.type == Dtype.null:
        return None
    buffer = _download_value(value.data)
    if value.type in [
        Dtype.float16, Dtype.float32, Dtype.float64,
        Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
        Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64,
        Dtype.bool
    ]:
        archive: NpzFile = npz_load(buffer)
        array = next(iter(archive.values()))
        return array if len(array.shape) else array.item()
    elif value.type == Dtype.string:
        return buffer.getvalue().decode("utf-8")
    elif value.type in [Dtype.list, Dtype.dict]:
        return loads(buffer.getvalue().decode("utf-8"))
    elif value.type == Dtype.image:
        return Image.open(buffer)
    elif value.type == Dtype.binary:
        return buffer
    else:
        raise ValueError(f"Failed to deserialize value with type `{value.type}` because it is not supported")

def _download_value(url: str) -> BytesIO:
    if url.startswith("data:"):
        with urlopen(url) as response:
            return BytesIO(response.read())
    response = get(url)
    response.raise_for_status()
    result = BytesIO(response.content)
    return result