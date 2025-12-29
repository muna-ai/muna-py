# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
from ..remote.remote import RemotePredictionService
from .completions import ChatCompletionService

class ChatService:
    """
    Chat service.
    """
    completions: ChatCompletionService

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.completions = ChatCompletionService(predictors, predictions, remote_predictions)