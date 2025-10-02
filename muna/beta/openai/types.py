# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel
from typing import Literal, TypedDict

class ChatCompletion(BaseModel):
    class Usage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage | None
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"] | None = None

class Message(BaseModel):
    role: Literal["assistant", "user", "system"]
    content: str | None = None

class _MessageDict(TypedDict): # For text completion
    role: Literal["assistant", "user", "system"]
    content: str | None

class DeltaMessage(BaseModel):
    role: Literal["assistant", "user", "system"] | None = None
    content: str | None = None

class Embedding(BaseModel):
    object: Literal["embedding"]
    embedding: list[float] | str
    index: int

class CreateEmbeddingResponse(BaseModel):
    class Usage(BaseModel):
        prompt_tokens: int
        total_tokens: int
    object: Literal["list"]
    model: str
    data: list[Embedding]
    usage: Usage