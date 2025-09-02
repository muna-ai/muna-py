# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ...services import PredictionService
from ..remote.remote import RemotePredictionService
from .completions import ChatCompletionsService

class ChatService:
    """
    Chat service.
    """
    completions: ChatCompletionsService

    def __init__(
        self,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.completions = ChatCompletionsService(predictions, remote_predictions)