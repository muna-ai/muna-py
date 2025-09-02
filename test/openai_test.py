# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from muna import Muna
from muna.beta.openai import ChatCompletionChunk, Message
from typing import Iterator

def test_create_chat_completion():
    muna = Muna()
    response = muna.beta.openai.chat.completions.create(
        model="@yusuf/llama-stream",
        messages=[
            { "role": "user", "content": "What is the capital of France?" },
            Message(role="user", content="And how many people live there?")
        ],
        stream=False
    )
    print(response.model_dump_json(indent=2))

def test_stream_chat_completion():
    muna = Muna()
    chunks = muna.beta.openai.chat.completions.create(
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