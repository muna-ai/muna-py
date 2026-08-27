#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from base64 import urlsafe_b64encode
from json import dumps
from pytest import fixture, raises

from muna.api import cache as cache_module
from muna.api import PredictionCache
from muna.services.prediction import PredictionService
from muna.types import Prediction

@fixture(autouse=True)
def _identity(monkeypatch):
    monkeypatch.setattr(cache_module, "Configuration", _FakeConfiguration)

def test_first_retrieve_fetches_and_persists():
    client = _FakeClient(lambda _: _prediction("first", iat=100, exp=1_100))
    cache = PredictionCache(client, now=lambda: 150)

    prediction = cache.retrieve("@test/model")

    assert prediction.id == "first"
    assert len(client.requests) == 1
    assert len(client.store) == 1


def test_fresh_token_is_served_from_cache():
    client = _FakeClient(lambda _: _prediction("first", iat=100, exp=1_100))
    cache = PredictionCache(client, now=lambda: 599)

    cache.retrieve("@test/model")
    prediction = cache.retrieve("@test/model")

    assert prediction.id == "first"
    assert len(client.requests) == 1


def test_half_life_refresh_is_pinned_to_cached_prediction():
    clock = { "now": 150 }
    client = _FakeClient(lambda _: _prediction("first", iat=100, exp=1_100))
    cache = PredictionCache(client, now=lambda: clock["now"])
    cache.retrieve("@test/model")

    clock["now"] = 600
    client.handler = lambda _: _prediction("second", iat=600, exp=1_600)
    prediction = cache.retrieve("@test/model")

    assert prediction.id == "second"
    assert len(client.requests) == 2
    assert client.requests[-1]["predictionId"] == "first"


def test_refresh_failure_falls_back_to_stale_prediction():
    clock = { "now": 150 }
    client = _FakeClient(lambda _: _prediction("first", iat=100, exp=1_100))
    cache = PredictionCache(client, now=lambda: clock["now"])
    cache.retrieve("@test/model")

    clock["now"] = 1_200
    client.handler = None
    prediction = cache.retrieve("@test/model")

    assert prediction.id == "first"


def test_invalidate_evicts_and_pins_next_fetch():
    client = _FakeClient(lambda _: _prediction("first", iat=100, exp=1_100))
    cache = PredictionCache(client, now=lambda: 150)
    cache.retrieve("@test/model")

    cache.invalidate("@test/model")

    assert len(client.store) == 0
    client.handler = lambda _: _prediction("second", iat=200, exp=1_200)
    prediction = cache.retrieve("@test/model")
    assert prediction.id == "second"
    assert client.requests[-1]["predictionId"] == "first"


def test_invalidated_prediction_is_never_reserved():
    client = _FakeClient(lambda _: _prediction("first", iat=100, exp=1_100))
    cache = PredictionCache(client, now=lambda: 150)
    cache.retrieve("@test/model")

    cache.invalidate("@test/model")
    client.handler = None

    with raises(RuntimeError):
        cache.retrieve("@test/model")


def test_raw_prediction_bypasses_cache():
    client = _FakeClient(lambda _: _prediction("raw", iat=100, exp=1_100))
    service = PredictionService(client)

    service.create(
        "@test/model",
        client_id="client",
        configuration_id="configuration"
    )

    assert len(client.requests) == 1
    assert len(client.store) == 0


def test_legacy_token_does_not_refresh():
    token = _token(iat=100)

    assert not PredictionCache._should_refresh_token(token, now=10_000)


def test_token_refreshes_at_half_life():
    token = _token(iat=100, exp=1_100)

    assert not PredictionCache._should_refresh_token(token, now=599)
    assert PredictionCache._should_refresh_token(token, now=600)


def test_malformed_token_refreshes():
    assert PredictionCache._should_refresh_token("not-a-token", now=0)


def test_cache_key_includes_client_and_configuration():
    key = PredictionCache._get_cache_key("tag", "client-a", "configuration-a")

    assert key != PredictionCache._get_cache_key("tag", "client-b", "configuration-a")
    assert key != PredictionCache._get_cache_key("tag", "client-a", "configuration-b")

def _token(*, iat: int | None=None, exp: int | None=None) -> str:
    claims = {
        key: value
        for key, value in { "iat": iat, "exp": exp }.items()
        if value is not None
    }
    payload = (
        urlsafe_b64encode(dumps(claims).encode())
        .decode()
        .rstrip("=")
    )
    return f"header.{payload}.signature"

def _prediction(id: str, *, iat: int, exp: int | None=None) -> Prediction:
    return Prediction(
        id=id,
        tag="@test/model",
        configuration=_token(iat=iat, exp=exp),
        resources=[],
        created="2026-08-27T00:00:00Z",
    )

class _FakeClient:

    def __init__(self, handler=None):
        self.handler = handler
        self.store: dict[str, str] = {}
        self.requests: list[dict] = []

    def request(self, *, method, path, body=None, response_type=None):
        self.requests.append(body or {})
        if self.handler is None:
            raise RuntimeError("offline")
        return self.handler(body)

    def get_cache_entry(self, key: str) -> str | None:
        return self.store.get(key)

    def set_cache_entry(self, key: str, value: str | None) -> None:
        if value is not None:
            self.store[key] = value
        else:
            self.store.pop(key, None)

    def download(self, url, path, *, progress=True):
        raise NotImplementedError

class _FakeConfiguration:

    @classmethod
    def get_client_id(cls) -> str:
        return "linux-x86_64"

    @classmethod
    def get_unique_id(cls) -> str:
        return "device-a"