# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from typing import Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Predictor, Prediction
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService
from .types import CreateEmbeddingResponse

class EmbeddingsService:
    """
    Embeddings service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__remote_predictions = remote_predictions
        self.__cache = dict[str, Predictor]()

    def create( # INCOMPLETE
        self,
        *,
        input: str | list[str],
        model: str,
        dimensions: int | None=None,
        encoding_format: Literal["float", "base64"] | None=None,
        acceleration: Acceleration | RemoteAcceleration="auto"
    ) -> CreateEmbeddingResponse:
        """
        Create an embedding vector representing the input text.

        Parameters:
            input (str | list): Input text to embed. The input must not exceed the max input tokens for the model.
            model (str): Embedding model predictor tag.
            dimensions (int): The number of dimensions the resulting output embeddings should have. Only supported by Matryoshka embedding models.
            encoding_format (str): The format to return the embeddings in.
            acceleration (Acceleration | RemoteAcceleration): Prediction acceleration.
        """
        encoding_format = encoding_format or "float"
        if model not in self.__cache:
            predictor = self.__predictors.retrieve(model)
            pass
        pass