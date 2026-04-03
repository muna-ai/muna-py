# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from typing import Iterator

from ..client import MunaClient
from ..types import Acceleration, Prediction, Value
from .local import LocalPredictionService
from .remote import RemotePredictionService

class PredictionService:
    """
    Make predictions.
    """

    def __init__(self, client: MunaClient):
        self.__local = LocalPredictionService(client)
        self.__remote = RemotePredictionService(client)

    def create(
        self,
        tag: str,
        *,
        inputs: dict[str, Value] | None=None,
        acceleration: Acceleration="local_auto",
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
        if inputs is None or acceleration.startswith("local_"):
            return self.__local.create(
                tag,
                inputs=inputs,
                acceleration=acceleration,
                device=device,
                client_id=client_id,
                configuration_id=configuration_id
            )
        else:
            return self.__remote.create(
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

        Parameters:
            tag (str): Predictor tag.
            inputs (dict): Input values.
            acceleration (Acceleration): Prediction acceleration.

        Returns:
            Iterator: Prediction stream.
        """
        if acceleration.startswith("local_"):
            return self.__local.stream(
                tag,
                inputs=inputs,
                acceleration=acceleration,
                device=device
            )
        else:
            return self.__remote.stream(
                tag,
                inputs=inputs,
                acceleration=acceleration
            )

    def delete(self, tag: str) -> bool:
        """
        Delete a predictor that is loaded in memory.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            bool: Whether the predictor was successfully deleted from memory.
        """
        return self.__local.delete(tag)