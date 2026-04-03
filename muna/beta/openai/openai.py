# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
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
        predictions: PredictionService
    ):
        self.chat = ChatService(predictors, predictions)
        self.embeddings = EmbeddingService(predictors, predictions)
        self.audio = AudioService(predictors, predictions)
        self.images = ImageService(predictors, predictions)