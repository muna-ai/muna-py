# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections import defaultdict
from typing import overload, Iterator, Literal
from typing_extensions import TypedDict

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device
from ..adapters.completions import (
    OpenAICompletionsAdapter, OpenAICompletionsOptions,
    ChatCompletion, ChatCompletionChunk,
    ChatCompletionFunctionTool, ChatCompletionMessageFunctionToolCall,
    ChatCompletionReasoningEffort, ChatCompletionToolChoice, Choice,
    Message, StreamChoice
)

class _ResponseFormatTextDict(TypedDict):
    type: Literal["text"]

class _ResponseFormatJsonSchemaDict(TypedDict):
    type: Literal["json_schema"]
    json_schema: dict[str, object]

_ResponseFormatDict = _ResponseFormatTextDict | _ResponseFormatJsonSchemaDict

class ChatCompletionService:
    """
    Create chat completions.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache: dict[str, OpenAICompletionsAdapter] = {}

    @overload
    def create(
        self,
        *,
        messages: list[Message],
        model: str,
        stream: Literal[False]=False,
        tools: list[ChatCompletionFunctionTool | dict] | None=None,
        tool_choice: ChatCompletionToolChoice | None=None,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        seed: int | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        messages: list[Message],
        model: str,
        stream: Literal[True],
        tools: list[ChatCompletionFunctionTool | dict] | None=None,
        tool_choice: ChatCompletionToolChoice | None=None,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        seed: int | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        messages: list[Message],
        model: str,
        stream: bool=False,
        tools: list[ChatCompletionFunctionTool | dict] | None=None,
        tool_choice: ChatCompletionToolChoice | None=None,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: ChatCompletionReasoningEffort | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        seed: int | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """
        Create a chat completion.

        Parameters:
            messages (list): Messages for the conversation so far.
            model (str): Chat model tag.
            stream (bool): Whether to stream responses.
            tools (list): Tools the model may call.
            tool_choice (ChatCompletionToolChoice): Tool choice mode. Defaults to `auto`.
            response_format (dict): Response format.
            reasoning_effort (ChatCompletionReasoningEffort): Reasoning effort for reasoning models.
            max_completion_tokens (int): Maximum completion tokens.
            temperature (float): Sampling temperature to use.
            top_p (float): Nucleus sampling coefficient.
            seed (int): Sampling seed for reproducible outputs.
            frequency_penalty (float): Token frequency penalty.
            presence_penalty (float): Token presence penalty.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.

        Returns:
            ChatCompletion | Iterator[ChatCompletionChunk]: Chat completion or chat completion chunks if streaming.
        """
        # Ensure adapter
        if model not in self.__cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            adapter = OpenAICompletionsAdapter.from_signature(predictor.signature)
            if adapter is None:
                raise ValueError(f"Predictor {model} does not support chat completions")
            self.__cache[model] = adapter
        # Predict
        adapter = self.__cache[model]
        inputs = adapter.inputs(
            messages=messages,
            tools=tools,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        predictions = self.__predictions.stream(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        # Adapt
        chunks = adapter.adapt(predictions, OpenAICompletionsOptions(include_usage=True))
        return chunks if stream else _merge_chunks(list(chunks))

def _merge_chunks(chunks: list[ChatCompletionChunk]) -> ChatCompletion:
    if not chunks:
        raise ValueError(f"Failed to parse chat completion because model did not return any outputs")
    choices_map = defaultdict[int, list[StreamChoice]](list)
    for chunk in chunks:
        for choice in chunk.choices:
            choices_map[choice.index].append(choice)
    choices = [
        _create_chat_completion_choice(index, choices)
        for index, choices in choices_map.items()
    ]
    chunk_usages = [chunk.usage for chunk in chunks if chunk.usage is not None]
    usage = ChatCompletion.Usage(
        prompt_tokens=sum(usage.prompt_tokens for usage in chunk_usages),
        completion_tokens=sum(usage.completion_tokens for usage in chunk_usages),
        total_tokens=sum(usage.total_tokens for usage in chunk_usages),
        # Engines report token details (e.g. cached / reasoning tokens) on the
        # final usage-bearing chunk; sums would double-count, so take the last.
        prompt_tokens_details=next((
            usage.prompt_tokens_details
            for usage in reversed(chunk_usages)
            if usage.prompt_tokens_details is not None
        ), None),
        completion_tokens_details=next((
            usage.completion_tokens_details
            for usage in reversed(chunk_usages)
            if usage.completion_tokens_details is not None
        ), None)
    )
    completion = ChatCompletion(
        id=chunks[0].id,
        created=chunks[0].created,
        model=chunks[0].model,
        choices=choices,
        usage=usage
    )
    return completion

def _create_chat_completion_choice(
    index: int,
    choices: list[StreamChoice]
) -> Choice:
    role = next((
        choice.delta.role
        for choice in choices
        if choice.delta is not None and choice.delta.role is not None
    ), "assistant")
    content = "".join(
        choice.delta.content
        for choice in choices
        if choice.delta and choice.delta.content
    )
    reasoning_content = "".join(
        choice.delta.reasoning_content
        for choice in choices
        if choice.delta and choice.delta.reasoning_content
    )
    message = ChatCompletion.Message(
        role=role,
        content=content,
        reasoning_content=reasoning_content if reasoning_content else None,
        tool_calls=_merge_tool_calls(choices)
    )
    finish_reason = next((
        choice.finish_reason
        for choice in choices
        if choice.finish_reason
    ), None)
    result = Choice(
        index=index,
        message=message,
        finish_reason=finish_reason
    )
    return result

def _merge_tool_calls(choices: list[StreamChoice]) -> list[ChatCompletionMessageFunctionToolCall] | None:
    """
    Accumulate streamed tool call fragments into completed tool calls,
    keyed by fragment index: the first fragment carries the id and
    function name; subsequent fragments append argument text.
    """
    calls: dict[int, dict] = {}
    for choice in choices:
        if choice.delta is None or not choice.delta.tool_calls:
            continue
        for fragment in choice.delta.tool_calls:
            call = calls.setdefault(fragment.index, { "id": "", "name": "", "arguments": "" })
            if fragment.id:
                call["id"] = fragment.id
            if fragment.function and fragment.function.name:
                call["name"] = fragment.function.name
            if fragment.function and fragment.function.arguments:
                call["arguments"] += fragment.function.arguments
    if not calls:
        return None
    return [ChatCompletionMessageFunctionToolCall(
        id=call["id"],
        function=ChatCompletionMessageFunctionToolCall.Function(
            name=call["name"],
            arguments=call["arguments"]
        )
    ) for _, call in sorted(calls.items())]
