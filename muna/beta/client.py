# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from functools import wraps
from inspect import signature as get_signature, Signature
from typing import get_origin, Callable, Generator, Iterator, TypeVar

from ..client import MunaClient
from ..services import PredictionService as EdgePredictionService
from ..types import Acceleration
from .openai import OpenAIService
from .remote import PredictionService, RemoteAcceleration

F = TypeVar("F", bound=Callable[..., object])

class BetaClient:
    """
    Client for incubating features.
    """
    predictions: PredictionService
    openai: OpenAIService
    
    def __init__(
        self,
        client: MunaClient,
        *,
        predictions: EdgePredictionService
    ):
        self.predictions = PredictionService(client)
        self.openai = OpenAIService(predictions, self.predictions.remote)
        self.__edge_predictions = predictions

    def predict( # INCOMPLETE # Preload
        self,
        tag: str,
        *,
        remote: bool=False,
        acceleration: Acceleration | RemoteAcceleration="auto",
        preload: bool=True
    ) -> Callable[[F], F]:
        """
        Create a prediction and return results when the decorated function is invoked.

        Parameters:
            tag (str): Predictor tag.
            remote (bool): Whether to create the prediction remotely.
            acceleration (Acceleration | RemoteAcceleration): Prediction acceleration.
            preload (bool): Whether to preload the predictor on the first run.
        """
        def decorator(func: F) -> F:
            signature = get_signature(func)
            @wraps(func)
            def wrapper(*args, **kwargs):
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                stream = (
                    signature.return_annotation is not Signature.empty and
                    get_origin(signature.return_annotation) in [Iterator, Generator]
                )
                create_func = self.predictions.remote.create if remote else self.__edge_predictions.create
                def _predict (): # INCOMPLETE
                    prediction = create_func(
                        tag=tag,
                        inputs=bound_args.arguments,
                        acceleration=acceleration
                    )
                    if prediction.error:
                        raise RuntimeError(prediction.error)
                    return tuple(prediction.results) if len(prediction.results) > 1 else prediction.results[0]
                result = _predict()
                return result
            return wrapper
        return decorator