# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import Muna
from muna.beta.anthropic import Message, RawContentBlockDeltaEvent, TextDelta
from typing import Iterator

MODEL = "@huggingface/smollm2-135m"

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
