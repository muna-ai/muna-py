# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from muna import Muna
from muna.beta.openai import ChatCompletionChunk, Message
from typing import Iterator

def test_create_chat_completion():
    openai = Muna().beta.openai
    response = openai.chat.completions.create(
        model="@openai/gpt-oss-20b",
        messages=[
            { "role": "user", "content": "What is the capital of France?" },
            Message(role="user", content="And how many people live there?")
        ],
        stream=False
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
        response_format="pcm",
        acceleration="auto"
    )
    assert response