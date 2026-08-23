# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ..client import MunaClient
from ..services import PredictorService, PredictionService
from .anthropic import AnthropicClient
from .deployments import DeploymentService
from .openai import OpenAIClient

class BetaClient:
    """
    Client for incubating features.
    """
    anthropic: AnthropicClient
    openai: OpenAIClient
    
    def __init__(
        self,
        client: MunaClient,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.anthropic = AnthropicClient(predictors, predictions)
        self.deployments = DeploymentService(client)
        self.openai = OpenAIClient(predictors, predictions)