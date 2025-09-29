# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from os import environ

from .beta.client import BetaClient
from .client import MunaClient
from .services import PredictionService, PredictorService, UserService

class Muna:
    """
    Muna client.

    Members:
        client (MunaClient): Muna API client. Do NOT use this unless you know what you are doing.
        users (UserService): Manage users.
        predictors (PredictorService): Manage predictors.
        predictions (PredictionService): Manage predictions.
        beta (BetaClient): Beta client for incubating features.

    Constructor:
        access_key (str): Muna access key.
        api_url (str): Muna API URL.
    """
    client: MunaClient
    users: UserService
    predictors: PredictorService
    predictions: PredictionService
    beta: BetaClient

    def __init__(
        self,
        access_key: str=None,
        *,
        api_url: str=None
    ):
        access_key = access_key or environ.get("MUNA_ACCESS_KEY") or environ.get("FXN_ACCESS_KEY")
        api_url = api_url or environ.get("MUNA_API_URL") or environ.get("FXN_API_URL")
        self.client = MunaClient(access_key, api_url)
        self.users = UserService(self.client)
        self.predictors = PredictorService(self.client)
        self.predictions = PredictionService(self.client)
        self.beta = BetaClient(self.client, self.predictors, self.predictions)