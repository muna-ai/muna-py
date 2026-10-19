# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from ..api import parse_hf_tag, load_hf_manifest, MunaClient, MunaAPIError
from ..types import Predictor, User

class PredictorService:

    def __init__(self, client: MunaClient) -> None:
        self.client = client

    def retrieve(self, tag: str) -> Predictor | None:
        """
        Retrieve a predictor.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            Predictor: Predictor.
        """
        if hf_tag := parse_hf_tag(tag):
            hf_manifest = load_hf_manifest(hf_tag)
            username, name = hf_manifest.tag.lstrip("@").split("/")
            return Predictor(
                tag=hf_manifest.tag,
                owner=User(username=username),
                name=name,
                description="",
                status="active",
                access="unlisted",
                signature=hf_manifest.signature,
                created=""
            )
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