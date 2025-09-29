# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from ..client import MunaClient
from ..services import PredictorService, PredictionService as EdgePredictionService
from .openai import OpenAIClient
from .remote import PredictionService

class BetaClient:
    """
    Client for incubating features.
    """
    predictions: PredictionService
    openai: OpenAIClient
    
    def __init__(
        self,
        client: MunaClient,
        predictors: PredictorService,
        predictions: EdgePredictionService
    ):
        self.predictions = PredictionService(client)
        self.openai = OpenAIClient(predictors, predictions, self.predictions.remote)