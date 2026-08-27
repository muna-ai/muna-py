#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from base64 import b64decode
from collections.abc import Callable
from json import loads
from pathlib import Path
from pydantic import ValidationError
from time import time
from urllib.parse import urlparse

from ..c import Configuration
from ..types import Prediction, PredictionResource
from .client import MunaClient, _get_cache_dir

class PredictionCache:
    """
    Prediction cache.
    This caches predictions used to provision local predictors, along with
    their configuration tokens and resources. Configuration tokens are
    refreshed once they fall within half of their lifetime, providing the
    liveness signal used to count monthly active runtimes.
    """

    def __init__(
        self,
        client: MunaClient,
        *,
        now: Callable[[], float]=time
    ):
        """
        Create the prediction cache.

        Parameters:
            client (MunaClient): Muna API client.
            now (callable): Clock returning the current Unix timestamp in seconds. Used for testing.
        """
        self.__client = client
        self.__now = now
        self.__identity: tuple[str, str] | None = None
        self.__invalidated_prediction_ids: dict[str, str] = {}

    def retrieve(self, tag: str) -> Prediction:
        """
        Retrieve a cached prediction.
        This serves the cached prediction when its configuration token is
        within half of its lifetime, and fetches a prediction from the API
        otherwise, falling back to the cached prediction when the API is
        unreachable. Prediction resources in the returned prediction always
        point to local files.

        Parameters:
            tag (str): Predictor tag.

        Returns:
            Prediction: Cached prediction.
        """
        client_id, configuration_id = self.__resolve_identity()
        key = self._get_cache_key(tag, client_id, configuration_id)
        prediction = self.__load_prediction(key)
        if (
            prediction is not None and
            prediction.configuration and
            not self._should_refresh_token(prediction.configuration, self.__now())
        ):
            return self.__localize(prediction)
        try:
            refreshed = self.__fetch_prediction(
                tag,
                client_id=client_id,
                configuration_id=configuration_id,
                key=key,
                cached_prediction_id=prediction.id if prediction is not None else None
            )
            return self.__localize(refreshed)
        except Exception:
            # Configuration token expiry is a refresh hint. A failed refresh
            # must not prevent an already-provisioned device from working offline.
            if prediction is None:
                raise
            return self.__localize(prediction)

    def invalidate(self, tag: str) -> None:
        """
        Invalidate a cached prediction.
        The evicted prediction identifier is remembered so that the next
        retrieval is pinned to it, ensuring that the fetched configuration
        token corresponds to the same predictor implementation.

        Parameters:
            tag (str): Predictor tag.
        """
        client_id, configuration_id = self.__resolve_identity()
        key = self._get_cache_key(tag, client_id, configuration_id)
        prediction = self.__load_prediction(key)
        if prediction is not None and prediction.id:
            self.__invalidated_prediction_ids[key] = prediction.id
        self.__client.set_cache_entry(key, None)

    def __resolve_identity(self) -> tuple[str, str]:
        if self.__identity is None:
            self.__identity = (
                Configuration.get_client_id(),
                Configuration.get_unique_id()
            )
        return self.__identity

    def __fetch_prediction(
        self,
        tag: str,
        *,
        client_id: str,
        configuration_id: str,
        key: str,
        cached_prediction_id: str | None
    ) -> Prediction:
        prediction_id = (
            self.__invalidated_prediction_ids.get(key) or
            cached_prediction_id
        )
        body: dict[str, object] = {
            "tag": tag,
            "clientId": client_id,
            "configurationId": configuration_id,
        }
        if prediction_id:
            body["predictionId"] = prediction_id
        prediction = self.__client.request(
            method="POST",
            path="/predictions",
            body=body,
            response_type=Prediction
        )
        self.__client.set_cache_entry(key, prediction.model_dump_json())
        self.__invalidated_prediction_ids.pop(key, None)
        return prediction

    def __load_prediction(self, key: str) -> Prediction | None:
        data = self.__client.get_cache_entry(key)
        if not data:
            return None
        try:
            return Prediction.model_validate_json(data)
        except ValidationError:
            self.__client.set_cache_entry(key, None)
            return None

    def __localize(self, prediction: Prediction) -> Prediction:
        resources = [
            self.__download_resource(resource)
            for resource in prediction.resources or []
        ]
        return prediction.model_copy(update={ "resources": resources })

    def __download_resource(
        self,
        resource: PredictionResource
    ) -> PredictionResource:
        if urlparse(resource.url).scheme not in ("http", "https"):
            return resource
        stem = Path(urlparse(resource.url).path).name
        path = _get_cache_dir() / stem
        path = path / resource.name if resource.name else path
        if not path.exists():
            color = "dark_orange" if resource.type != "dso" else "purple"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.__client.download(resource.url, path, progress=color)
        return resource.model_copy(update={ "url": str(path) })

    @staticmethod
    def _should_refresh_token(token: str, now: float) -> bool:
        claims = parse_configuration_claims(token)
        if claims is None:
            # Revalidate malformed cache entries, while still allowing stale
            # fallback if the refresh cannot reach the API.
            return True
        exp = claims.get("exp")
        if not isinstance(exp, (int, float)) or isinstance(exp, bool):
            return False # Legacy tokens do not expire.
        iat = claims.get("iat")
        if (
            not isinstance(iat, (int, float)) or
            isinstance(iat, bool) or
            exp <= iat
        ):
            return now >= exp
        refresh_at = iat + (exp - iat) / 2
        return now >= refresh_at

    @staticmethod
    def _get_cache_key(
        tag: str,
        client_id: str,
        configuration_id: str
    ) -> str:
        encoded = "".join(
            f"{len(value)}:{value}"
            for value in (tag, client_id, configuration_id)
        )
        return f"muna.prediction.{encoded}"

def parse_configuration_claims(configuration_token: str | None) -> dict[str, object] | None:
    if configuration_token is None:
        return None
    try:
        payload = configuration_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = loads(b64decode(payload, altchars=b"-_").decode("utf-8"))
        return claims if isinstance(claims, dict) else None
    except (AttributeError, IndexError, TypeError, ValueError):
        return None
