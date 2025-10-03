# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
from ..remote.remote import RemotePredictionService
from .audio import AudioService
from .chat import ChatService
from .embeddings import EmbeddingService

class OpenAIClient:
    """
    Experimental client mimicking the official OpenAI client.

    Members:
        chat (ChatService): Chat service.
        embeddings (EmbeddingsService): Embeddings service.
    """
    chat: ChatService
    embeddings: EmbeddingService
    audio: AudioService

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.chat = ChatService(predictors, predictions, remote_predictions)
        self.embeddings = EmbeddingService(predictors, predictions, remote_predictions)
        self.audio = AudioService(predictors, predictions, remote_predictions)