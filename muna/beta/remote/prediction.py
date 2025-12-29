# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ...client import MunaClient
from .remote import RemotePredictionService

class PredictionService:
    """
    Make predictions.
    """
    remote: RemotePredictionService

    def __init__(self, client: MunaClient):
        self.remote = RemotePredictionService(client)