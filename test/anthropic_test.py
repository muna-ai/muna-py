# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import Muna
from muna.beta.anthropic import Message, RawContentBlockDeltaEvent, TextDelta
from muna.beta.anthropic.models import _create_model_info, _unix_to_rfc3339
from muna.beta.openai.models import _create_model
from muna.beta.utils import rfc3339_to_unix, split_tag
from muna.types import Signature
from typing import Iterator

MODEL = "@huggingface/smollm2-135m"

def _chat_signature() -> Signature:
    return Signature.model_validate({
        "inputs": [
            {
                "name": "messages",
                "dtype": "list",
                "denotation": "openai.chat.completions.messages",
                "contextLength": 262144
            },
            {
                "name": "tools",
                "dtype": "list",
                "denotation": "openai.chat.completions.tools",
                "optional": True
            },
            {
                "name": "reasoning_effort",
                "dtype": "string",
                "denotation": "openai.chat.completions.reasoning_effort",
                "optional": True,
                "enumeration": [
                    { "name": "none", "value": "none" },
                    { "name": "low", "value": "low" },
                    { "name": "medium", "value": "medium" },
                    { "name": "high", "value": "high" },
                    { "name": "xhigh", "value": "xhigh" }
                ]
            },
            {
                "name": "max_output_tokens",
                "dtype": "int32",
                "denotation": "openai.chat.completions.max_output_tokens",
                "optional": True,
                "min": 1,
                "max": 65536
            },
            {
                "name": "temperature",
                "dtype": "float32",
                "denotation": "openai.chat.completions.temperature",
                "optional": True
            }
        ],
        "outputs": []
    })

def test_chat_model_reports_limits_and_capabilities():
    info = _create_model_info("@qwen/qwen-3.8-27b", _chat_signature(), created=1_789_936_000)
    assert info.type == "model"
    assert info.display_name == "qwen-3.8-27b"
    assert info.created_at == "2026-09-20T20:26:40Z"
    assert info.max_input_tokens == 262_144
    assert info.max_tokens == 65_536
    caps = info.capabilities
    assert caps.thinking.supported
    assert caps.thinking.types.enabled.supported
    assert not caps.thinking.types.adaptive.supported
    assert caps.effort.supported
    assert caps.effort.low.supported and caps.effort.xhigh.supported
    assert not caps.effort.max.supported
    assert not caps.image_input.supported
    assert not caps.structured_outputs.supported
    data = info.model_dump()
    assert data["type"] == "model"
    assert data["capabilities"]["effort"]["high"]["supported"]

def test_signature_without_chat_denotations_is_a_bare_model():
    signature = Signature.model_validate({
        "inputs": [{ "name": "image", "dtype": "image" }],
        "outputs": [{ "name": "label", "dtype": "string" }]
    })
    info = _create_model_info("@acme/classifier", signature, created=None)
    assert info.created_at == "1970-01-01T00:00:00Z"
    assert not info.capabilities.thinking.supported
    assert not info.capabilities.effort.supported
    # Limits are nullable and always present.
    data = info.model_dump()
    assert data["max_input_tokens"] is None
    assert data["max_tokens"] is None

def test_free_text_effort_advertises_every_level():
    signature = Signature.model_validate({
        "inputs": [
            { "name": "messages", "dtype": "list" },
            {
                "name": "effort",
                "dtype": "string",
                "denotation": "openai.chat.completions.reasoning_effort"
            },
            {
                "name": "images",
                "dtype": "list",
                "denotation": "openai.chat.completions.images"
            }
        ],
        "outputs": []
    })
    info = _create_model_info("@a/b", signature, created=None)
    assert info.capabilities.effort.max.supported
    assert info.capabilities.image_input.supported

def test_projections_agree_with_the_openai_model():
    # The two dialects read the same signature independently; a server
    # flattens both into one row, so they must agree wherever they overlap.
    signature = _chat_signature()
    info = _create_model_info("@qwen/qwen-3.8-27b", signature, created=1_789_936_000)
    model = _create_model("@qwen/qwen-3.8-27b", signature, created=1_789_936_000)
    assert info.id == model.id
    assert rfc3339_to_unix(info.created_at) == model.created
    assert f"@{model.owned_by}/{info.display_name}" == model.id
    shared = model.model_dump().keys() & info.model_dump().keys()
    assert shared == { "id" }

def test_create_message():
    anthropic = Muna().beta.anthropic
    message = anthropic.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{ "role": "user", "content": "What is the capital of France?" }],
        acceleration="local_auto"
    )
    assert isinstance(message, Message)
    assert message.type == "message"
    assert message.role == "assistant"
    assert any(block.type == "text" and block.text for block in message.content)
    assert message.stop_reason in ("end_turn", "max_tokens")
    print(message.model_dump_json(indent=2))

def test_stream_message():
    anthropic = Muna().beta.anthropic
    events = anthropic.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{ "role": "user", "content": "What is the capital of France?" }],
        stream=True,
        acceleration="local_auto"
    )
    assert isinstance(events, Iterator)
    events = list(events)
    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"
    assert events[-2].type == "message_delta"
    text = "".join(
        event.delta.text for event in events
        if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, TextDelta)
    )
    assert text

def test_stream_helper():
    anthropic = Muna().beta.anthropic
    with anthropic.messages.stream(
        model=MODEL,
        max_tokens=1024,
        messages=[{ "role": "user", "content": "What is the capital of France?" }],
        acceleration="local_auto"
    ) as stream:
        text = "".join(stream.text_stream)
        message = stream.get_final_message()
    assert text
    assert isinstance(message, Message)
    assert any(block.type == "text" and block.text for block in message.content)
