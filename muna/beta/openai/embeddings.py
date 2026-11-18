# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from typing import Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device
from ..adapters import (
    EmbeddingCreateResponse, OpenAIEmbeddingsAdapter,
    OpenAIEmbeddingsOptions
)

class EmbeddingService:
    """
    Embedding service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache: dict[str, OpenAIEmbeddingsAdapter] = {}

    def create(
        self,
        *,
        input: str | list[str],
        model: str,
        dimensions: int | None=None,
        encoding_format: Literal["float", "base64"] | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> EmbeddingCreateResponse:
        """
        Create an embedding vector representing the input text.

        Parameters:
            input (str | list): Input text to embed. The input must not exceed the max input tokens for the model.
            model (str): Embedding model tag.
            dimensions (int): The number of dimensions the resulting output embeddings should have. Only supported by Matryoshka embedding models.
            encoding_format (str): The format to return the embeddings in.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.
        """
        # Ensure adapter
        if model not in self.__cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            adapter = OpenAIEmbeddingsAdapter.from_signature(predictor.signature)
            if adapter is None:
                raise ValueError(f"Predictor {model} does not support embeddings")
            self.__cache[model] = adapter
        # Create prediction
        adapter = self.__cache[model]
        inputs = adapter.inputs(input=input, dimensions=dimensions)
        prediction = self.__predictions.create(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        # Adapt
        options = OpenAIEmbeddingsOptions(encoding_format=encoding_format or "float")
        response = adapter.adapt(prediction, options)
        # Return
        return response
