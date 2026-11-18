#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from muna.beta import Annotations
from muna.beta.openai import ChatCompletionChunk, Message
from typing import Annotated, Iterator

@compile(targets=["x86_64-unknown-linux-gnu"])
def chat(
    messages: Annotated[list[Message], Annotations.ChatMessages(
        description="Messages in the conversation.",
        context_length=4096
    )],
    *,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature.",
        min=0.,
        max=2.,
    )] = 0.7,
    max_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum generated tokens.",
        min=1,
        max=4096
    )] = 128
) -> Iterator[ChatCompletionChunk]:
    """
    Signature-only chat completion predictor.
    """
    raise NotImplementedError("Signature-only mock predictor.")
