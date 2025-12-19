#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ctypes import c_uint8, cast, string_at, POINTER
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from io import BytesIO
from json import loads
from numpy import array, dtype, generic, ndarray
from numpy.ctypeslib import as_array, as_ctypes_type
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from tempfile import gettempdir
from typing import Iterator
from urllib.parse import urlparse

from ..c import (
    Configuration, Predictor, Prediction as LocalPrediction,
    Value as LocalValue, ValueMap, ValueFlags
)
from ..client import MunaClient
from ..resources import download_resource
from ..types import Acceleration, Dtype, Prediction, PredictionResource, Value

class PredictionService:

    def __init__(self, client: MunaClient):
        self.client = client
        self.__cache = dict[str, Predictor]()
        self.__cache_dir = _get_home_dir() / ".fxn" / "cache"
        self.__cache_dir.mkdir(parents=True, exist_ok=True)

    def __del__(self):
        while self.__cache:
            _, predictor = self.__cache.popitem()
            with predictor:
                pass

    def create(
        self,
        tag: str,
        *,
        inputs: dict[str, Value] | None=None,
        acceleration: Acceleration="auto",
        device=None,
        client_id: str=None,
        configuration_id: str=None
    ) -> Prediction:
        """
        Create a prediction.

        Parameters:
            tag (str): Predictor tag.
            inputs (dict): Input values.
            acceleration (Acceleration): Prediction acceleration.
            client_id (str): Muna client identifier. Specify this to override the current client identifier.
            configuration_id (str): Configuration identifier. Specify this to override the current client configuration identifier.

        Returns:
            Prediction: Created prediction.
        """
        if inputs is None:
            return self.__create_raw_prediction(
                tag=tag,
                client_id=client_id,
                configuration_id=configuration_id
            )
        predictor = self.__get_predictor(
            tag=tag,
            acceleration=acceleration,
            device=device,
            client_id=client_id,
            configuration_id=configuration_id
        )
        with (
            _create_value_map(inputs) as input_map,
            predictor.create_prediction(input_map) as prediction
        ):
            return _parse_local_prediction(prediction, tag=tag)

    def stream(
        self,
        tag: str,
        *,
        inputs: dict[str, Value],
        acceleration: Acceleration="auto",
        device=None
    ) -> Iterator[Prediction]:
        """
        Stream a prediction.

        Parameters:
            tag (str): Predictor tag.
            inputs (dict): Input values.
            acceleration (Acceleration): Prediction acceleration.

        Returns:
            Iterator: Prediction stream.
        """
        predictor = self.__get_predictor(
            tag=tag,
            acceleration=acceleration,
            device=device,
        )
        with (
            _create_value_map(inputs) as input_map,
            predictor.stream_prediction(input_map) as stream
        ):
            for prediction in stream:
                with prediction:
                    yield _parse_local_prediction(prediction, tag=tag)

    def delete(self, tag: str) -> bool:
        """
        Delete a predictor that is loaded in memory.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            bool: Whether the predictor was successfully deleted from memory.
        """
        if tag not in self.__cache:
            return False
        with self.__cache.pop(tag):
            return True

    def __create_raw_prediction(
        self,
        tag: str,
        client_id: str=None,
        configuration_id: str=None
    ) -> Prediction:
        client_id = client_id if client_id is not None else Configuration.get_client_id()
        configuration_id = configuration_id if configuration_id is not None else Configuration.get_unique_id()
        prediction = self.client.request(
            method="POST",
            path="/predictions",
            body={
                "tag": tag,
                "clientId": client_id,
                "configurationId": configuration_id,
            },
            response_type=Prediction
        )
        return prediction

    def __get_predictor(
        self,
        tag: str,
        acceleration: Acceleration="auto",
        device=None,
        client_id: str=None,
        configuration_id: str=None
    ) -> Predictor:
        if tag in self.__cache:
            return self.__cache[tag]
        prediction = self.__create_raw_prediction(
            tag=tag,
            client_id=client_id,
            configuration_id=configuration_id
        )
        with Configuration() as configuration:
            configuration.tag = prediction.tag
            configuration.token = prediction.configuration
            configuration.acceleration = acceleration
            configuration.device = device
            for resource in prediction.resources:
                path = self.__get_resource_path(resource)
                if not path.exists():
                    color = "dark_orange" if not resource.type == "dso" else "purple"
                    download_resource(resource.url, path, progress=color)
                configuration.add_resource(resource.type, path)
            predictor = Predictor(configuration)
        self.__cache[tag] = predictor
        return predictor

    def __get_resource_path(self, resource: PredictionResource) -> Path:
        stem = Path(urlparse(resource.url).path).name
        path = self.__cache_dir / stem
        path = path / resource.name if resource.name else path
        return path

def _create_value_map(inputs: dict[str, Value]) -> ValueMap:
    map = ValueMap()
    for name, value in inputs.items():
        map[name] = _create_local_value(value)
    return map

def _create_local_value(
    obj: Value,
    *,
    flags: ValueFlags=ValueFlags.NONE
) -> LocalValue:
    obj = _ensure_object_serializable(obj)
    match obj:
        case None:          return LocalValue.create_null()
        case float():       return _create_local_value(array(obj, dtype=Dtype.float32), flags=flags | ValueFlags.COPY_DATA)
        case bool():        return _create_local_value(array(obj, dtype=Dtype.bool), flags=flags | ValueFlags.COPY_DATA)
        case int():         return _create_local_value(array(obj, dtype=Dtype.int32), flags=flags | ValueFlags.COPY_DATA)
        case generic():     return _create_local_value(array(obj), flags=flags | ValueFlags.COPY_DATA)
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

def _parse_local_prediction(
    prediction: LocalPrediction,
    *,
    tag: str
) -> Prediction:
    output_map = prediction.results
    results: list[Value] | None = None
    if output_map:
        output_values = [output_map[output_map.key(idx)] for idx in range(len(output_map))]
        results = [_parse_local_value(value) for value in output_values]
    return Prediction(
        id=prediction.id,
        tag=tag,
        results=results,
        latency=prediction.latency,
        error=prediction.error,
        logs=prediction.logs,
        created=datetime.now(timezone.utc).isoformat()
    )

def _parse_local_value(value: LocalValue) -> Value:
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

def _ensure_object_serializable(obj: object) -> object:
    is_dict = is_dataclass(obj) and not isinstance(obj, type)
    match obj:
        case list():        return list(map(_ensure_object_serializable, obj))
        case BaseModel():   return obj.model_dump(mode="json", by_alias=True)
        case _ if is_dict:  return asdict(obj)
        case _:             return obj

def _get_home_dir() -> Path:
    try:
        check = Path.home() / ".fxntest"
        with open(check, "w") as f:
            f.write("fxn")
        check.unlink()
        return Path.home()
    except:
        return Path(gettempdir())
    
_TENSOR_DTYPES = {
    Dtype.bfloat16, Dtype.float16, Dtype.float32, Dtype.float64,
    Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
    Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64,
    Dtype.bool,
}
_JSON_TYPES = { Dtype.list, Dtype.dict }