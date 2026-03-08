# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
from ..remote.remote import RemotePredictionService
from .audio import AudioService
from .chat import ChatService
from .embeddings import EmbeddingService
from .images import ImageService

class OpenAIClient:
    """
    Experimental client mimicking the official OpenAI client.

    Members:
        chat (ChatService): Chat service.
        embeddings (EmbeddingsService): Embeddings service.
        audio (AudioService): Audio service.
        image (ImageService): Image service.
    """
    chat: ChatService
    embeddings: EmbeddingService
    audio: AudioService
    images: ImageService

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.chat = ChatService(predictors, predictions, remote_predictions)
        self.embeddings = EmbeddingService(predictors, predictions, remote_predictions)
        self.audio = AudioService(predictors, predictions, remote_predictions)
        self.images = ImageService(predictors, predictions, remote_predictions)