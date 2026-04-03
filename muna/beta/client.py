# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ..client import MunaClient
from ..services import PredictorService, PredictionService
from .openai import OpenAIClient

class BetaClient:
    """
    Client for incubating features.
    """
    openai: OpenAIClient
    
    def __init__(
        self,
        client: MunaClient,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.openai = OpenAIClient(predictors, predictions)