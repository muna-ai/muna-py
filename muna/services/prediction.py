#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Iterator
from urllib.parse import urlparse

from ..c import Configuration, Predictor, Prediction as CPrediction, ValueMap
from ..client import MunaClient
from ..resources import download_resource
from ..types import Acceleration, Prediction, PredictionResource, Value

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
            ValueMap.from_dict(inputs) as input_map,
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
            ValueMap.from_dict(inputs) as input_map,
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

def _parse_local_prediction(
    prediction: CPrediction,
    *,
    tag: str
) -> Prediction:
    output_map = prediction.results
    results: list[Value] | None = None
    if output_map:
        output_values = [output_map[output_map.key(idx)] for idx in range(len(output_map))]
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

def _get_home_dir() -> Path:
    try:
        check = Path.home() / ".fxntest"
        with open(check, "w") as f:
            f.write("fxn")
        check.unlink()
        return Path.home()
    except:
        return Path(gettempdir())