# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from base64 import b64encode
from io import BytesIO
from muna import Muna
from muna.beta import Annotations
from muna.beta.openai import (
    ChatCompletionChunk, ChoiceDeltaToolCall, DeltaMessage, Message,
    StreamChoice
)
from muna.beta.openai.completions import _merge_tool_calls, _normalize_conversation
from pathlib import Path
from PIL import Image
from pytest import raises
from typing import Iterator

def test_normalize_string_content_passes_through():
    conversation = _normalize_conversation(
        [
            { "role": "system", "content": "You are a helpful assistant." },
            Message(role="user", content="What is the capital of France?")
        ],
        images_param=None,
        audios_param=None
    )
    assert [m.content for m in conversation.messages] == [
        "You are a helpful assistant.",
        "What is the capital of France?"
    ]
    assert conversation.images == []
    assert conversation.audios == []

def test_normalize_text_parts_flatten_with_newline():
    conversation = _normalize_conversation(
        [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "line one" },
                    { "type": "text", "text": "line two" }
                ]
            }
        ],
        images_param=None,
        audios_param=None
    )
    assert conversation.messages[0].content == "line one\nline two"

def test_normalize_refusal_flattens_as_text():
    conversation = _normalize_conversation(
        [
            {
                "role": "assistant",
                "content": [
                    { "type": "refusal", "refusal": "I cannot help with that." }
                ]
            }
        ],
        images_param=None,
        audios_param=None
    )
    assert conversation.messages[0].content == "I cannot help with that."

def test_normalize_image_parts_decode_in_order_across_messages():
    images_param = Annotations.ChatImages(description="Images.")
    conversation = _normalize_conversation(
        [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "first" },
                    {
                        "type": "image_url",
                        "image_url": { "url": _png_data_url((255, 0, 0, 255)) }
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": { "url": _png_data_url((0, 255, 0, 255)) }
                    }
                ]
            }
        ],
        images_param=images_param,
        audios_param=None
    )
    # The nth placeholder across the conversation indexes the nth image.
    assert len(conversation.images) == 2
    assert conversation.images[0].getpixel((0, 0)) == (255, 0, 0, 255)
    assert conversation.images[1].getpixel((0, 0)) == (0, 255, 0, 255)
    first = conversation.messages[0].content
    assert [part.type for part in first] == ["text", "image"]
    second = conversation.messages[1].content
    assert [part.type for part in second] == ["image"]

def test_normalize_undeclared_image_raises():
    with raises(ValueError, match="image_url"):
        _normalize_conversation(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": { "url": _png_data_url((0, 0, 255, 255)) }
                        }
                    ]
                }
            ],
            images_param=None,
            audios_param=None
        )

def test_normalize_file_part_raises():
    with raises(ValueError, match="File content parts"):
        _normalize_conversation(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": { "file_data": "AAAA", "filename": "doc.pdf" }
                        }
                    ]
                }
            ],
            images_param=None,
            audios_param=None
        )

def test_normalize_tool_turns_pass_through():
    conversation = _normalize_conversation(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": { "name": "get_weather", "arguments": "{\"city\": \"Paris\"}" }
                    }
                ]
            },
            { "role": "tool", "content": "18C and sunny", "tool_call_id": "call_123" }
        ],
        images_param=None,
        audios_param=None
    )
    assistant, tool = conversation.messages
    assert assistant.tool_calls[0].function.name == "get_weather"
    assert assistant.content is None
    assert tool.role == "tool"
    assert tool.tool_call_id == "call_123"

def test_merge_tool_call_fragments():
    choices = [
        StreamChoice(
            index=0,
            delta=DeltaMessage(
                role="assistant",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_test_0",
                        type="function",
                        function=ChoiceDeltaToolCall.Function(name="get_weather", arguments="")
                    )
                ]
            )
        ),
        StreamChoice(
            index=0,
            delta=DeltaMessage(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCall.Function(arguments="{\"location\": ")
                    )
                ]
            )
        ),
        StreamChoice(
            index=0,
            delta=DeltaMessage(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCall.Function(arguments="\"Paris\"}")
                    )
                ]
            )
        ),
        StreamChoice(index=0, delta=DeltaMessage(), finish_reason="tool_calls")
    ]
    calls = _merge_tool_calls(choices)
    assert len(calls) == 1
    assert calls[0].id == "call_test_0"
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == "{\"location\": \"Paris\"}"

def test_merge_tool_calls_absent_returns_none():
    choices = [StreamChoice(index=0, delta=DeltaMessage(content="hi"), finish_reason="stop")]
    assert _merge_tool_calls(choices) is None

def test_create_chat_completion():
    openai = Muna().beta.openai
    response = openai.chat.completions.create(
        model="@openai/gpt-oss-20b",
        messages=[
            { "role": "user", "content": "What is the capital of France?" },
            Message(role="user", content="And how many people live there?")
        ],
        stream=False,
        acceleration="local_auto"
    )
    print(response.model_dump_json(indent=2))

def test_stream_chat_completion():
    openai = Muna().beta.openai
    chunks = openai.chat.completions.create(
        model="@openai/gpt-oss-20b",
        messages=[
            { "role": "user", "content": "What is the capital of France?" },
            Message(role="user", content="And how many people live there?")
        ],
        stream=True,
        acceleration="local_auto"
    )
    assert(isinstance(chunks, Iterator))
    for chunk in chunks:
        assert isinstance(chunk, ChatCompletionChunk)

def test_create_embedding():
    openai = Muna().beta.openai
    response = openai.embeddings.create(
        input="Hello world",
        model="@google/embedding-gemma"
    )
    assert response.object == "list"
    assert len(response.data) > 0
    assert response.data[0].object == "embedding"
    assert isinstance(response.data[0].embedding, list) and isinstance(response.data[0].embedding[0], float)

def test_create_embedding_base64():
    openai = Muna().beta.openai
    response = openai.embeddings.create(
        input="Hello world",
        model="@google/embedding-gemma",
        encoding_format="base64"
    )
    assert response.object == "list"
    assert len(response.data) > 0
    assert response.data[0].object == "embedding"
    assert isinstance(response.data[0].embedding, str)

def test_create_speech():
    openai = Muna().beta.openai
    response = openai.audio.speech.create(
        input="Hello from Muna",
        model="@kitten-ml/kitten-tts",
        voice="expr-voice-2-f",
        response_format="mp3",
        acceleration="local_auto"
    )
    assert response

def test_create_transcription():
    openai = Muna().beta.openai
    audio_path = Path("test/data/librispeech_sample.wav")
    with audio_path.open("rb") as f:
        transcription = openai.audio.transcriptions.create(
            file=f,
            model="@moonshine/moonshine-base",
        )
    assert transcription.text.lower().startswith("going along slushy country roads")

def _png_data_url(color: tuple[int, int, int, int]) -> str:
    buffer = BytesIO()
    Image.new("RGBA", (4, 4), color).save(buffer, format="PNG")
    return f"data:image/png;base64,{b64encode(buffer.getvalue()).decode()}"