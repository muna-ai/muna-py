#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel
from typing import Iterator, Literal

from ...c import Configuration
from ...client import MunaClient
from ...services._utils import create_remote_value, parse_remote_value
from ...types import Prediction, Value
from .schema import RemoteAcceleration, RemotePrediction

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
        input_map = { name: create_remote_value(value).model_dump(mode="json") for name, value in inputs.items() }
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
        input_map = { name: create_remote_value(value).model_dump(mode="json") for name, value in inputs.items() }
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

def _parse_remote_prediction(prediction: RemotePrediction) -> Prediction:
    """
    Download a remote prediction.
    """
    # Download results
    results = (
        list(map(parse_remote_value, prediction.results))
        if prediction.results is not None
        else None
    )
    # Return
    return Prediction(
        id=prediction.id,
        tag=prediction.tag,
        results=results,
        latency=prediction.latency,
        error=prediction.error,
        logs=prediction.logs,
        created=prediction.created
    )

class _RemotePredictionEvent(BaseModel):
    event: Literal["prediction"]
    data: RemotePrediction