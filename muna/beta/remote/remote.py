#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from base64 import b64encode
from io import BytesIO
from json import dumps, loads
from numpy import array, generic, load as npz_load, ndarray, savez_compressed
from numpy.lib.npyio import NpzFile
from PIL import Image
from pydantic import BaseModel
from requests import get
from typing import Iterator, Literal
from urllib.request import urlopen

from ...c import Configuration
from ...c.value import _ensure_object_serializable, _TENSOR_DTYPES
from ...client import MunaClient
from ...types import Dtype, Prediction, Value
from .schema import RemoteAcceleration, RemotePrediction, RemoteValue

class RemotePredictionService:
    """
    Make remote predictions.
    """

    def __init__(self, client: MunaClient):
        self.client = client

    def create(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: RemoteAcceleration="remote_auto"
    ) -> Prediction:
        """
        Create a remote prediction.

        Parameters:
            tag (str): Predictor tag.
            inputs (dict): Input values.
            acceleration (RemoteAcceleration): Prediction acceleration.

        Returns:
            Prediction: Created prediction.
        """
        input_map = { name: _create_remote_value(value).model_dump(mode="json") for name, value in inputs.items() }
        remote_prediction = self.client.request(
            method="POST",
            path="/predictions/remote",
            body={
                "tag": tag,
                "inputs": input_map,
                "acceleration": acceleration,
                "clientId": Configuration.get_client_id()
            },
            response_type=RemotePrediction
        )
        prediction = _parse_remote_prediction(remote_prediction)
        return prediction

    def stream(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: RemoteAcceleration="remote_auto"
    ) -> Iterator[Prediction]:
        """
        Stream a remote prediction.

        Parameters:
            tag (str): Predictor tag.
            inputs (dict): Input values.
            acceleration (Acceleration): Prediction acceleration.

        Returns:
            Iterator: Prediction stream.
        """
        input_map = { name: _create_remote_value(value).model_dump(mode="json") for name, value in inputs.items() }
        for event in self.client.stream(
            method="POST",
            path=f"/predictions/remote",
            body={
                "tag": tag,
                "inputs": input_map,
                "acceleration": acceleration,
                "clientId": Configuration.get_client_id(),
                "stream": True
            },
            response_type=_RemotePredictionEvent
        ):
            prediction = _parse_remote_prediction(event.data)
            yield prediction

def _create_remote_value(obj: Value) -> RemoteValue:
    obj = _ensure_object_serializable(obj)
    match obj:
        case None:      return RemoteValue(data=None, dtype=Dtype.null)
        case float():   return _create_remote_value(array(obj, dtype=Dtype.float32))
        case bool():    return _create_remote_value(array(obj, dtype=Dtype.bool))
        case int():     return _create_remote_value(array(obj, dtype=Dtype.int32))
        case generic(): return _create_remote_value(array(obj))
        case ndarray():
            buffer = BytesIO()
            savez_compressed(buffer, obj, allow_pickle=False)
            data = _upload_value_data(buffer)
            return RemoteValue(data=data, dtype=obj.dtype.name)
        case str():
            buffer = BytesIO(obj.encode())
            data = _upload_value_data(buffer, mime="text/plain")
            return RemoteValue(data=data, dtype=Dtype.string)
        case list():
            buffer = BytesIO(dumps(obj).encode())
            data = _upload_value_data(buffer, mime="application/json")
            return RemoteValue(data=data, dtype=Dtype.list)
        case dict():
            buffer = BytesIO(dumps(obj).encode())
            data = _upload_value_data(buffer, mime="application/json")
            return RemoteValue(data=data, dtype=Dtype.dict)
        case Image.Image():
            buffer = BytesIO()
            format = "PNG" if obj.mode == "RGBA" else "JPEG"
            mime = f"image/{format.lower()}"
            obj.save(buffer, format=format)
            data = _upload_value_data(buffer, mime=mime)
            return RemoteValue(data=data, dtype=Dtype.image)
        case BytesIO():
            data = _upload_value_data(obj)
            return RemoteValue(data=data, dtype=Dtype.binary)
        case _:
            raise ValueError(f"Failed to serialize value '{obj}' of type `{type(obj)}` because it is not supported")

def _parse_remote_value(value: RemoteValue) -> Value:
    buffer = _download_value_data(value.data) if value.data else None
    is_tensor = value.dtype in _TENSOR_DTYPES
    match value.dtype:
        case Dtype.null:
            return None
        case _ if is_tensor:
            archive: NpzFile = npz_load(buffer)
            array = next(iter(archive.values()))
            return array if len(array.shape) else array.item()
        case Dtype.string:
            return buffer.getvalue().decode("utf-8")
        case Dtype.list | Dtype.dict:
            return loads(buffer.getvalue().decode("utf-8"))
        case Dtype.image:
            return Image.open(buffer)
        case Dtype.binary:
            return buffer
        case _:
            raise ValueError(f"Failed to parse remote value with type `{value.dtype}` because it is not supported")

def _parse_remote_prediction(prediction: RemotePrediction) -> Prediction:
    results = (
        list(map(_parse_remote_value, prediction.results))
        if prediction.results is not None
        else None
    )
    return Prediction(
        id=prediction.id,
        tag=prediction.tag,
        results=results,
        latency=prediction.latency,
        error=prediction.error,
        logs=prediction.logs,
        created=prediction.created
    )

def _upload_value_data(
    data: BytesIO,
    *,
    mime: str="application/octet-stream"
) -> str:
    encoded_data = b64encode(data.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded_data}"

def _download_value_data(url: str) -> BytesIO:
    if url.startswith("data:"):
        with urlopen(url) as response:
            return BytesIO(response.read())
    response = get(url)
    response.raise_for_status()
    result = BytesIO(response.content)
    return result

class _RemotePredictionEvent(BaseModel):
    event: Literal["prediction"]
    data: RemotePrediction