#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from base64 import b64encode
from datetime import datetime, timezone
from io import BytesIO
from json import dumps, loads
from numpy import array, generic, ndarray, savez_compressed
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError
from requests import get
from typing import Iterator, Literal
from urllib.request import urlopen

from ..api import parse_configuration_claims, MunaClient, PredictionCache
from ..c import Configuration, Predictor, Prediction as CPrediction, ValueMap
from ..c.value import _ensure_object_serializable, Value as CValue, _TENSOR_DTYPES
from ..types import (
    Acceleration, Dtype, Prediction,
    RemotePrediction, RemoteValue, Value
)

class PredictionService:
    """
    Make local and remote predictions.
    """

    def __init__(self, client: MunaClient):
        self.client = client
        self.cache = PredictionCache(client)
        self.__predictors = dict[str, Predictor]()

    def __del__(self):
        while self.__predictors:
            _, predictor = self.__predictors.popitem()
            with predictor:
                pass

    def create(
        self,
        tag: str,
        *,
        inputs: dict[str, Value] | None=None,
        acceleration: Acceleration="local_auto",
        device=None,
        client_id: str | None=None,
        configuration_id: str | None=None
    ) -> Prediction:
        """
        Create a prediction.
        """
        if inputs is None:
            return self.__create_raw_prediction(
                tag,
                client_id=(
                    client_id
                    if client_id is not None
                    else Configuration.get_client_id()
                ),
                configuration_id=(
                    configuration_id
                    if configuration_id is not None
                    else Configuration.get_unique_id()
                )
            )
        if not inputs or acceleration.startswith("local_"):
            return self.__create_local_prediction(
                tag,
                inputs=inputs,
                acceleration=acceleration,
                device=device
            )
        return self.__create_remote_prediction(
            tag,
            inputs=inputs,
            acceleration=acceleration
        )

    def stream(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: Acceleration="local_auto",
        device=None
    ) -> Iterator[Prediction]:
        """
        Stream a prediction.
        """
        if acceleration.startswith("local_"):
            yield from self.__stream_local_prediction(
                tag,
                inputs=inputs,
                acceleration=acceleration,
                device=device
            )
        else:
            yield from self.__stream_remote_prediction(
                tag,
                inputs=inputs,
                acceleration=acceleration
            )

    def delete(self, tag: str) -> bool:
        """
        Delete a predictor that is loaded in memory.
        """
        predictor = self.__predictors.pop(tag, None)
        if predictor is None:
            return False
        with predictor:
            pass
        return True

    def __create_local_prediction(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: Acceleration,
        device=None
    ) -> Prediction:
        if not inputs:
            return self.cache.retrieve(tag)
        predictor = self.__get_predictor(
            tag,
            acceleration=acceleration,
            device=device
        )
        with (
            ValueMap.from_dict(inputs) as input_map,
            predictor.create_prediction(input_map) as prediction
        ):
            return _parse_local_prediction(prediction, tag=tag)

    def __stream_local_prediction(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: Acceleration,
        device=None
    ) -> Iterator[Prediction]:
        predictor = self.__get_predictor(
            tag=tag,
            acceleration=acceleration,
            device=device
        )
        with (
            ValueMap.from_dict(inputs) as input_map,
            predictor.stream_prediction(input_map) as stream
        ):
            for prediction in stream:
                with prediction:
                    yield _parse_local_prediction(prediction, tag=tag)

    def __create_remote_prediction(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: Acceleration
    ) -> Prediction:
        input_map = {
            name: _create_remote_value(value).model_dump(mode="json")
            for name, value in inputs.items()
        }
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
        return _parse_remote_prediction(remote_prediction)

    def __stream_remote_prediction(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: Acceleration
    ) -> Iterator[Prediction]:
        input_map = {
            name: _create_remote_value(value).model_dump(mode="json")
            for name, value in inputs.items()
        }
        for event in self.client.stream(
            method="POST",
            path="/predictions/remote",
            body={
                "tag": tag,
                "inputs": input_map,
                "acceleration": acceleration,
                "clientId": Configuration.get_client_id(),
                "stream": True
            },
            response_type=_RemotePredictionEvent
        ):
            yield _parse_remote_prediction(event.data)

    def __create_raw_prediction(
        self,
        tag: str,
        *,
        client_id: str,
        configuration_id: str
    ) -> Prediction:
        return self.client.request(
            method="POST",
            path="/predictions",
            body={
                "tag": tag,
                "clientId": client_id,
                "configurationId": configuration_id,
            },
            response_type=Prediction
        )

    def __get_predictor(
        self,
        tag: str,
        *,
        acceleration: Acceleration="local_auto",
        device: object | None=None
    ) -> Predictor:
        if tag in self.__predictors:
            return self.__predictors[tag]
        prediction = self.cache.retrieve(tag)
        try:
            predictor = self.__create_predictor(
                prediction,
                acceleration=acceleration,
                device=device
            )
        except _PredictorCreationError:
            # A cached token can outlive native validation rules. Evict it and
            # retry once with a fetch pinned to the evicted prediction.
            self.cache.invalidate(tag)
            prediction = self.cache.retrieve(tag)
            predictor = self.__create_predictor(
                prediction,
                acceleration=acceleration,
                device=device
            )
        self.__predictors[tag] = predictor
        return predictor

    def __create_predictor(
        self,
        prediction: Prediction,
        *,
        acceleration: Acceleration,
        device: object
    ) -> Predictor:
        with Configuration() as configuration:
            configuration.tag = prediction.tag
            configuration.token = prediction.configuration
            configuration.acceleration = acceleration
            configuration.devices = [] if device is None else [device]
            for resource in prediction.resources:
                configuration.add_resource(resource.type, resource.url)
            for entry in _parse_preload_claim(prediction.configuration):
                preload_prediction = self.create(
                    entry.tag,
                    inputs={ "_": None },
                    acceleration=entry.acceleration
                )
                configuration.set_metadata(
                    entry.metadata,
                    _preload_output(preload_prediction, entry.tag)
                )
            try:
                return Predictor(configuration)
            except RuntimeError as error:
                raise _PredictorCreationError(str(error)) from error

def _parse_preload_claim(configuration_token: str | None) -> list[_PreloadEntry]:
    claims = parse_configuration_claims(configuration_token)
    if claims is None:
        return []
    try:
        preload = claims.get("preload", [])
        if not isinstance(preload, list):
            return []
        result: list[_PreloadEntry] = []
        for entry in preload:
            try:
                result.append(_PreloadEntry.model_validate(entry))
            except ValidationError:
                continue
        return result
    except (TypeError, ValueError):
        return []

def _preload_output(prediction: Prediction, tag: str) -> str:
    if prediction.error:
        raise RuntimeError(f"Failed to preload {tag}: {prediction.error}")
    if not prediction.results:
        raise RuntimeError(
            f"Failed to preload {tag} because it returned no results"
        )
    result = prediction.results[0]
    if not isinstance(result, str):
        raise RuntimeError(
            f"Failed to preload {tag} because its first result is not a string"
        )
    return result

def _parse_local_prediction(
    prediction: CPrediction,
    *,
    tag: str
) -> Prediction:
    output_map = prediction.results
    results: list[Value] | None = None
    if output_map:
        output_values = [
            output_map[output_map.key(index)]
            for index in range(len(output_map))
        ]
        results = [value.to_object() for value in output_values]
    return Prediction(
        id=prediction.id,
        tag=tag,
        results=results,
        latency=prediction.latency,
        error=prediction.error,
        logs=prediction.logs,
        created=datetime.now(timezone.utc).isoformat()
    )

def _create_remote_value(obj: Value | RemoteValue) -> RemoteValue:
    if isinstance(obj, RemoteValue):
        return obj
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
        case list() if all(isinstance(tensor, ndarray) for tensor in obj):
            buffer = BytesIO()
            savez_compressed(buffer, *obj, allow_pickle=False)
            data = _upload_value_data(buffer, mime="application/x-npz")
            return RemoteValue(data=data, dtype=Dtype.array_list)
        case list() if all(isinstance(image, Image.Image) for image in obj):
            value = CValue.from_object(obj)
            buffer = BytesIO(value.serialize(mime="image/avif"))
            data = _upload_value_data(buffer, mime="image/avif")
            return RemoteValue(data=data, dtype=Dtype.image_list)
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
            raise ValueError(
                f"Failed to serialize value '{obj}' of type `{type(obj)}` "
                "because it is not supported"
            )

def _parse_remote_value(value: RemoteValue) -> Value:
    buffer = _download_value_data(value.data) if value.data else None
    is_tensor = value.dtype in _TENSOR_DTYPES
    match value.dtype:
        case Dtype.null:
            return None
        case _ if is_tensor:
            c_value = CValue.from_bytes(
                buffer.getvalue(),
                mime="application/vnd.muna.tensor"
            )
            return c_value.to_object()
        case Dtype.string:
            return buffer.getvalue().decode("utf-8")
        case Dtype.list | Dtype.dict:
            return loads(buffer.getvalue().decode("utf-8"))
        case Dtype.image:
            return Image.open(buffer)
        case Dtype.array_list:
            c_value = CValue.from_bytes(
                buffer.getvalue(),
                mime="application/x-npz"
            )
            return c_value.to_object()
        case Dtype.image_list:
            c_value = CValue.from_bytes(
                buffer.getvalue(),
                mime="image/avif"
            )
            return c_value.to_object()
        case Dtype.binary:
            return buffer
        case _:
            raise ValueError(
                f"Failed to parse remote value with type `{value.dtype}` "
                "because it is not supported"
            )

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
    return BytesIO(response.content)

class _RemotePredictionEvent(BaseModel):
    event: Literal["prediction"]
    data: RemotePrediction

class _PreloadEntry(BaseModel, **ConfigDict(frozen=True)):
    tag: str
    acceleration: str
    metadata: str

class _PredictorCreationError(RuntimeError):
    pass