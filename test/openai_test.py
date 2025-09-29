# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from muna import Muna
from muna.beta.openai import ChatCompletionChunk, Message
from typing import Iterator

from openai import OpenAI

def test_create_chat_completion():
    openai = Muna().beta.openai
    response = openai.chat.completions.create(
        model="@yusuf/llama-stream",
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
        model="@yusuf/llama-stream",
        messages=[
            { "role": "user", "content": "What is the capital of France?" },
            Message(role="user", content="And how many people live there?")
        ],
        stream=True
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
    pass