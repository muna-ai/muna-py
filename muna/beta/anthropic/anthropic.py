# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ...services import PredictorService, PredictionService
from .messages import MessageService

class AnthropicClient:
    """
    Experimental client mimicking the official Anthropic client.

    Members:
        messages (MessageService): Messages service.
    """
    messages: MessageService

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.messages = MessageService(predictors, predictions)