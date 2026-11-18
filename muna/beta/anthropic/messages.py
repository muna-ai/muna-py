# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Iterator
from json import loads
from typing import overload, Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device

from ..adapters.messages import (
    AnthropicMessagesAdapter,
    Message, RawContentBlockDeltaEvent, RawContentBlockStartEvent,
    RawContentBlockStopEvent, RawMessageDeltaEvent, RawMessageStartEvent,
    RawMessageStopEvent, RawMessageStreamEvent, StopReason, TextBlock,
    TextDelta, ThinkingBlock, ThinkingDelta, Usage, _MessageParamDict,
    _TextBlockParamDict, InputJSONDelta, ToolParam, ToolUseBlock
)

class MessageService:
    """
    Message service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache: dict[str, AnthropicMessagesAdapter] = {}

    @overload
    def create(
        self,
        *,
        max_tokens: int,
        messages: list[_MessageParamDict],
        model: str,
        stop_sequences: list[str] | None=None,
        stream: Literal[False]=False,
        system: str | list[_TextBlockParamDict] | None=None,
        temperature: float | None=None,
        top_k: int | None=None,
        top_p: float | None=None,
        tools: list[ToolParam] | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> Message: ...

    @overload
    def create(
        self,
        *,
        max_tokens: int,
        messages: list[_MessageParamDict],
        model: str,
        stop_sequences: list[str] | None=None,
        stream: Literal[True],
        system: str | list[_TextBlockParamDict] | None=None,
        temperature: float | None=None,
        top_k: int | None=None,
        top_p: float | None=None,
        tools: list[ToolParam] | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> Iterator[RawMessageStreamEvent]: ...

    def create(
        self,
        *,
        max_tokens: int,
        messages: list[_MessageParamDict],
        model: str,
        stop_sequences: list[str] | None=None,
        stream: bool=False,
        system: str | list[_TextBlockParamDict] | None=None,
        temperature: float | None=None,
        top_k: int | None=None,
        top_p: float | None=None,
        tools: list[ToolParam] | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> Message | Iterator[RawMessageStreamEvent]:
        """
        Create a message.

        Parameters:
            max_tokens (int): The maximum number of tokens to generate before stopping.
            messages (list): Input messages.
            model (str): Model predictor tag.
            stop_sequences (list): Custom text sequences that will cause the model to stop generating.
            stream (bool): Whether to incrementally stream the response.
            system (str | list): System prompt.
            temperature (float): Amount of randomness injected into the response.
            top_k (int): Only sample from the top K options for each subsequent token.
            top_p (float): Nucleus sampling coefficient.
            tools (list): Tools the model may call.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.

        Returns:
            Message | Iterator[RawMessageStreamEvent]: Message or raw message stream events if streaming.
        """
        # Ensure adapter
        if model not in self.__cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            adapter = AnthropicMessagesAdapter.from_signature(predictor.signature)
            if adapter is None:
                raise ValueError(f"Predictor {model} does not support Anthropic messages")
            self.__cache[model] = adapter
        # Predict
        adapter = self.__cache[model]
        inputs = adapter.inputs(
            max_tokens=max_tokens,
            messages=messages,
            system=system,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tools=tools
        )
        predictions = self.__predictions.stream(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        # Adapt
        events = adapter.adapt(predictions)
        return events if stream else MessageStream(events).get_final_message()

    def stream(
        self,
        *,
        max_tokens: int,
        messages: list[_MessageParamDict],
        model: str,
        stop_sequences: list[str] | None=None,
        system: str | list[_TextBlockParamDict] | None=None,
        temperature: float | None=None,
        top_k: int | None=None,
        top_p: float | None=None,
        tools: list[ToolParam] | None=None,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> MessageStream:
        """
        Create a streaming message.

        Parameters:
            max_tokens (int): The maximum number of tokens to generate before stopping.
            messages (list): Input messages.
            model (str): Model predictor tag.
            stop_sequences (list): Custom text sequences that will cause the model to stop generating.
            system (str | list): System prompt.
            temperature (float): Amount of randomness injected into the response.
            top_k (int): Only sample from the top K options for each subsequent token.
            top_p (float): Nucleus sampling coefficient.
            tools (list): Tools the model may call.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.

        Returns:
            MessageStream: Message stream context manager.
        """
        events = self.create(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            stop_sequences=stop_sequences,
            stream=True,
            system=system,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tools=tools,
            acceleration=acceleration,
            devices=devices
        )
        return MessageStream(events)

class MessageStream:
    """
    Message stream context manager.
    """

    def __init__(self, events: Iterator[RawMessageStreamEvent]):
        self.__events = events
        self.__message: Message | None = None
        self.__tool_json: dict[int, str] = {}

    def __enter__(self) -> MessageStream:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        close = getattr(self.__events, "close", None)
        if close is not None:
            close()
        return False

    def __iter__(self) -> Iterator[RawMessageStreamEvent]:
        for event in self.__events:
            self.__accumulate(event)
            yield event

    @property
    def text_stream(self) -> Iterator[str]:
        """
        Iterator over the text deltas in the stream.
        """
        for event in self:
            match event:
                case RawContentBlockDeltaEvent(delta=TextDelta(text=text)):
                    yield text

    def get_final_message(self) -> Message:
        """
        Drain the stream and return the accumulated final message.
        """
        for _ in self:
            pass
        if self.__message is None:
            raise ValueError(
                "Failed to create message because the model "
                "did not return any outputs"
            )
        return self.__message

    def __accumulate(self, event: RawMessageStreamEvent):
        match event:
            case RawMessageStartEvent():
                self.__message = event.message.model_copy(deep=True)
            case RawContentBlockStartEvent() if self.__message is not None:
                self.__message.content.append(event.content_block.model_copy(deep=True))
            case RawContentBlockDeltaEvent() if self.__message is not None:
                block = self.__message.content[event.index]
                if isinstance(event.delta, TextDelta) and isinstance(block, TextBlock):
                    block.text += event.delta.text
                elif isinstance(event.delta, ThinkingDelta) and isinstance(block, ThinkingBlock):
                    block.thinking += event.delta.thinking
                elif isinstance(event.delta, InputJSONDelta) and isinstance(block, ToolUseBlock):
                    self.__tool_json[event.index] = (
                        self.__tool_json.get(event.index, "") + event.delta.partial_json
                    )
            case RawContentBlockStopEvent() if self.__message is not None:
                data = self.__tool_json.pop(event.index, None)
                if data is not None:
                    value = loads(data)
                    if not isinstance(value, dict):
                        raise ValueError("Tool input must be a JSON object")
                    self.__message.content[event.index].input = value
            case RawMessageDeltaEvent() if self.__message is not None:
                self.__message.stop_reason = event.delta.stop_reason
                self.__message.stop_sequence = event.delta.stop_sequence
                usage = event.usage
                if usage.input_tokens is not None:
                    self.__message.usage.input_tokens = usage.input_tokens
                self.__message.usage.output_tokens = usage.output_tokens
                self.__message.usage.cache_creation_input_tokens = usage.cache_creation_input_tokens
                self.__message.usage.cache_read_input_tokens = usage.cache_read_input_tokens
            case _:
                pass
