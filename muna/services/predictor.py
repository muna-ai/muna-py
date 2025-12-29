# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ..client import MunaClient, MunaAPIError
from ..types import Predictor

class PredictorService:

    def __init__(self, client: MunaClient) -> None:
        self.client = client

    def retrieve(self, tag: str) -> Predictor:
        """
        Retrieve a predictor.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            Predictor: Predictor.
        """
        try:
            return self.client.request(
                method="GET",
                path=f"/predictors/{tag}",
                response_type=Predictor
            )
        except MunaAPIError as error:
            if error.status_code == 404:
                return None
            raise