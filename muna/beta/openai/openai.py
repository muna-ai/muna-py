# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ...services import PredictionService
from ..remote.remote import RemotePredictionService
from .chat import ChatService
from .embeddings import EmbeddingsService

class OpenAIClient:
    """
    Experimental client mimicking the official OpenAI client.

    Members:
        chat (ChatService): Chat service.
    """
    chat: ChatService
    embeddings: EmbeddingsService

    def __init__(
        self,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.chat = ChatService(predictions, remote_predictions)
        self.embeddings = EmbeddingsService(predictions, remote_predictions)