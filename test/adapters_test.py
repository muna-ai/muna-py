#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from importlib import import_module
from pytest import mark, raises
from types import SimpleNamespace
from unittest.mock import Mock

from muna.beta.adapters import (
    AnthropicMessagesAdapter, OpenAICompletionsAdapter, OpenAIDepthAdapter,
    OpenAIEmbeddingsAdapter, OpenAIImageEditAdapter, OpenAIImageGenerationAdapter,
    OpenAISpeechAdapter, OpenAITranscriptionsAdapter, TypeSafeSystemOneAdapter,
)
from muna.beta.adapters._utils import get_parameter
from muna.beta.openai.embeddings import EmbeddingService
from muna.types import Parameter, Prediction, Signature

CASES = [
    (
        OpenAICompletionsAdapter,
        {
            "inputs": [
                {
                    "name": "messages",
                    "dtype": "list",
                    "denotation": "openai.chat.completions.messages"
                }
            ],
            "outputs": [
                {
                    "name": "chunk",
                    "dtype": "dict",
                    "schema": {
                        "title": "ChatCompletionChunk"
                    }
                }
            ]
        },
    ),
    (
        AnthropicMessagesAdapter,
        {
            "inputs": [
                {
                    "name": "messages",
                    "dtype": "list",
                    "denotation": "openai.chat.completions.messages"
                }
            ],
            "outputs": [
                {
                    "name": "chunk",
                    "dtype": "dict",
                    "schema": {
                        "title": "ChatCompletionChunk"
                    }
                }
            ]
        },
    ),
    (
        OpenAIEmbeddingsAdapter,
        {
            "inputs": [
                {
                    "name": "texts",
                    "dtype": "list"
                }
            ],
            "outputs": [
                {
                    "name": "vectors",
                    "dtype": "float32",
                    "denotation": "embedding"
                }
            ]
        },
    ),
    (
        OpenAIImageGenerationAdapter,
        {
            "inputs": [
                {
                    "name": "prompt",
                    "dtype": "string",
                    "denotation": "openai.images.prompt"
                }
            ],
            "outputs": [
                {
                    "name": "images",
                    "dtype": "image_list"
                }
            ]
        },
    ),
    (
        OpenAIImageEditAdapter,
        {
            "inputs": [
                {
                    "name": "prompt",
                    "dtype": "string",
                    "denotation": "openai.images.prompt"
                },
                {
                    "name": "images",
                    "dtype": "image_list",
                    "denotation": "openai.chat.completions.images"
                }
            ],
            "outputs": [
                {
                    "name": "images",
                    "dtype": "image_list"
                }
            ]
        },
    ),
    (
        OpenAIDepthAdapter,
        {
            "inputs": [
                {
                    "name": "prompt",
                    "dtype": "string",
                    "denotation": "openai.images.prompt"
                },
                {
                    "name": "images",
                    "dtype": "image_list",
                    "denotation": "openai.chat.completions.images"
                }
            ],
            "outputs": [
                {
                    "name": "depth",
                    "dtype": "array_list",
                    "denotation": "depth_map"
                }
            ]
        },
    ),
    (
        OpenAISpeechAdapter,
        {
            "inputs": [
                {
                    "name": "text",
                    "dtype": "string"
                },
                {
                    "name": "voice",
                    "dtype": "string",
                    "denotation": "openai.audio.speech.voice"
                }
            ],
            "outputs": [
                {
                    "name": "audio",
                    "dtype": "float32",
                    "denotation": "audio",
                    "sampleRate": 24000
                }
            ]
        },
    ),
    (
        OpenAITranscriptionsAdapter,
        {
            "inputs": [
                {
                    "name": "audio",
                    "dtype": "float32",
                    "denotation": "audio",
                    "sampleRate": 16000
                }
            ],
            "outputs": [
                {
                    "name": "text",
                    "dtype": "string"
                }
            ]
        },
    ),
    (
        TypeSafeSystemOneAdapter,
        {
            "inputs": [
                {
                    "name": "state",
                    "dtype": "list",
                    "denotation": "typesafe.systemone.state"
                },
                {
                    "name": "questions",
                    "dtype": "dict",
                    "denotation": "typesafe.systemone.questions"
                }
            ],
            "outputs": [
                {
                    "name": "response",
                    "dtype": "dict",
                    "schema": {
                        "title": "SystemOneResponse"
                    }
                }
            ]
        },
    ),
]

@mark.parametrize(
    "adapter_type,payload",
    CASES,
    ids=[adapter.__name__ for adapter, _ in CASES]
)
def test_signature_matching_is_optional_and_strict(adapter_type, payload):
    signature = Signature.model_validate(payload)
    original = signature.model_dump()
    adapter = adapter_type.from_signature(signature)
    assert adapter is not None
    assert signature.model_dump() == original
    assert adapter_type.from_signature(Signature(inputs=[], outputs=[])) is None

    missing_output = signature.model_copy(deep=True)
    missing_output.outputs = []
    assert adapter_type.from_signature(missing_output) is None

    wrong_type = signature.model_copy(deep=True)
    wrong_type.inputs[0].dtype = "bool"
    assert adapter_type.from_signature(wrong_type) is None

    unsupported = signature.model_copy(deep=True)
    unsupported.inputs.append(Parameter(name="unsupported", dtype="bool"))
    assert adapter_type.from_signature(unsupported) is None

    duplicate_name = signature.model_copy(deep=True)
    duplicate_name.inputs.append(Parameter(name=signature.inputs[0].name, dtype="bool", optional=True))
    assert adapter_type.from_signature(duplicate_name) is None

    ambiguous = signature.model_copy(deep=True)
    duplicate = ambiguous.inputs[0].model_copy(update={"name": "duplicate"})
    ambiguous.inputs.append(duplicate)
    assert adapter_type.from_signature(ambiguous) is None

    ambiguous_output = signature.model_copy(deep=True)
    ambiguous_output.outputs.append(ambiguous_output.outputs[0].model_copy(update={"name": "duplicate"}))
    assert adapter_type.from_signature(ambiguous_output) is None

    optional = signature.model_copy(deep=True)
    optional.inputs.append(Parameter(name="unused", dtype="bool", optional=True))
    assert adapter_type.from_signature(optional) is not None

@mark.parametrize(
    "adapter_type,payload",
    CASES,
    ids=[adapter.__name__ for adapter, _ in CASES]
)
def test_matching_does_not_hide_unexpected_errors(adapter_type, payload, monkeypatch):
    signature = Signature.model_validate(payload)
    module = import_module(adapter_type.__module__)
    def broken_lookup(*args, **kwargs):
        raise ValueError("Unexpected lookup bug")
    monkeypatch.setattr(module, "get_parameter", broken_lookup)
    with raises(ValueError, match="Unexpected lookup bug"):
        adapter_type.from_signature(signature)

def test_ambiguous_optional_parameter_is_omitted():
    signature = Signature.model_validate(CASES[0][1])
    signature.inputs.extend([
        Parameter(name=name, dtype="float32", optional=True, denotation="openai.chat.completions.temperature")
        for name in ("first", "second")
    ])
    for adapter_type in (OpenAICompletionsAdapter, AnthropicMessagesAdapter):
        adapter = adapter_type.from_signature(signature)
        assert adapter is not None
        assert adapter.temperature is None

def test_client_rejects_incompatible_signature_before_inference_and_caching():
    predictors = Mock()
    predictors.retrieve.return_value = SimpleNamespace(signature=Signature(inputs=[], outputs=[]))
    predictions = Mock()
    service = EmbeddingService(predictors, predictions)
    for _ in range(2):
        with raises(ValueError, match="Predictor @test/model does not support embeddings"):
            service.create(input="Hello", model="@test/model")
    assert predictors.retrieve.call_count == 2
    predictions.create.assert_not_called()

def test_adaptation_failures_still_raise():
    adapter = OpenAIEmbeddingsAdapter.from_signature(Signature.model_validate(CASES[2][1]))
    prediction = Prediction(id="pred_test", tag="@test/model", created="2026-11-17T00:00:00Z", results=[])
    with raises(ValueError, match="did not return"):
        adapter.adapt(prediction)

def test_parameter_lookup_returns_none_unless_exactly_one_parameter_matches():
    first = Parameter(name="first", dtype="string", denotation="prompt")
    second = Parameter(name="second", dtype="string", denotation="prompt")
    assert get_parameter([], dtype="string") == (None, None)
    assert get_parameter([first], dtype="string", denotation="other") == (None, None)
    assert get_parameter([first], dtype="float32") == (None, None)
    assert get_parameter([first], dtype="string", denotation="prompt") == (0, first)
    assert get_parameter([first, second], dtype="string", denotation="prompt") == (None, None)
