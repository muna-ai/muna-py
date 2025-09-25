# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ...services import PredictionService
from ..remote.remote import RemotePredictionService
from .completions import ChatCompletionsService

class EmbeddingsService:
    """
    Embeddings service.
    """

    def __init__(
        self,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.__predictions = predictions
        self.__remote_predictions = remote_predictions