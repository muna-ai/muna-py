# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
from .messages import MessageService
from .models import ModelService

class AnthropicClient:
    """
    Experimental client mimicking the official Anthropic client.

    Members:
        messages (MessageService): Messages service.
        models (ModelService): Model service.
    """
    messages: MessageService
    models: ModelService

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.messages = MessageService(predictors, predictions)
        self.models = ModelService(predictors)