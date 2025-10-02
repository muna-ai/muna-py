# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from base64 import b64encode
from collections.abc import Callable
from numpy import ndarray
from typing import Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService
from .types import CreateEmbeddingResponse, Embedding

PredictorEmbeddingDelegate = Callable[..., CreateEmbeddingResponse]

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
        self.__cache = dict[str, PredictorEmbeddingDelegate]()

    def create(
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
        input = [input] if isinstance(input, str) else input
        encoding_format = encoding_format or "float"
        # Ensure we have a delegate
        if model not in self.__cache:
            self.__cache[model] = self.__create_embedding_delegate(model)
        # Make prediction
        delegate = self.__cache[model]
        result = delegate(
            input=input,
            model=model,
            dimensions=dimensions,
            encoding_format=encoding_format,
            acceleration=acceleration
        )
        # Return
        return result

    def __create_embedding_delegate(self, tag: str) -> PredictorEmbeddingDelegate:
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        signature = predictor.signature
        # Check that there is only one required input parameter
        if sum(param for param in predictor.signature.inputs if not param.optional) != 1:
            raise ValueError(f"{tag} cannot be used with OpenAI embedding API because it has more than one required input parameter.")
        # Check that the input parameter is `list[str]`
        input_param = next(
            (param for param in signature.inputs if param.type == Dtype.list),
            None
        )
        if input_param is None:
            raise ValueError(f"{tag} cannot be used with OpenAI embedding API because it does not have a valid text embedding input parameter.")
        # Get the Matryoshka dim parameter (optional)
        matryoshka_param = next(
            (param for param in predictor.signature.inputs if "int" in param.type and param.denotation == "embedding.dims"),
            None
        )
        # Get the index of the embedding output
        embedding_param_idx = next(
            (idx for idx, param in enumerate(predictor.signature.outputs) if param.type == Dtype.float32 and param.denotation == "embedding"),
            None
        )
        if embedding_param_idx is None:
            raise ValueError(f"{tag} cannot be used with OpenAI embedding API because it has no outputs with an `embedding` denotation.")
        # Get the index of the usage output (optional)
        usage_param_idx = next(
            (idx for idx, param in enumerate(predictor.signature.outputs) if param.denotation == "openai.embedding.usage"),
            None
        )
        # Define delegate
        def create_embedding_func(
            *,
            input: list[str],
            model: str,
            dimensions: int | None,
            encoding_format: Literal["float", "base64"],
            acceleration: Acceleration | RemoteAcceleration
        ) -> CreateEmbeddingResponse:
            # Get prediction creation function (local or remote)
            create_prediction_func = (
                self.__remote_predictions.create
                if acceleration.startswith("remote_")
                else self.__predictions.create
            )
            # Build prediction input map
            prediction_inputs = { [input_param.name]: input }
            if matryoshka_param is not None:
                prediction_inputs[matryoshka_param.name] = dimensions
            # Create prediction
            prediction = create_prediction_func(
                tag=model,
                inputs=prediction_inputs,
                acceleration=acceleration
            )
            # Check for error
            if prediction.error:
                raise RuntimeError(prediction.error)
            # Create embedding response
            embedding_matrix: ndarray = prediction.results[embedding_param_idx] # (N,D)
            if len(embedding_matrix.shape) != 2:
                raise RuntimeError(f"{tag} returned embedding matrix with invalid shape: {embedding_matrix.shape}")
            usage = (
                CreateEmbeddingResponse.Usage(**prediction.results[usage_param_idx])
                if usage_param_idx is not None
                else CreateEmbeddingResponse.Usage(prompt_tokens=0, total_tokens=0)
            )
            embeddings = [self.__parse_embedding(
                embedding,
                index=idx,
                encoding_format=encoding_format
            ) for idx, embedding in enumerate(embedding_matrix)]
            response = CreateEmbeddingResponse(
                object="list",
                model=model,
                data=embeddings,
                usage=usage
            )
            # Return
            return response
        # Return
        return create_embedding_func

    def __parse_embedding(
        self,
        embedding_vector: ndarray,
        *,
        index: int,
        encoding_format: Literal["float", "base64"]
    ) -> Embedding:
        data = (
            b64encode(embedding_vector.tobytes()).decode()
            if encoding_format else
            embedding_vector.tolist()
        )
        embedding = Embedding(
            object="embedding",
            embedding=data,
            index=index
        )
        return embedding