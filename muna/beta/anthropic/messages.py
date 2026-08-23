# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Callable, Iterator
from pydantic import TypeAdapter, ValidationError
from typing import overload, Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype, Parameter, Prediction, Signature
from ..annotations import get_parameter
from ..openai.schema import ChatCompletionChunk, _MessageDict
from .schema import (
    Message, RawContentBlockDeltaEvent, RawContentBlockStartEvent,
    RawContentBlockStopEvent, RawMessageDeltaEvent, RawMessageStartEvent,
    RawMessageStopEvent, RawMessageStreamEvent, StopReason, TextBlock,
    TextDelta, ThinkingBlock, ThinkingDelta, Usage, _MessageParamDict,
    _TextBlockParamDict
)

MessageDelegate = Callable[..., Message | Iterator[RawMessageStreamEvent]]

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
        self.__cache = dict[str, MessageDelegate]()

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
        acceleration: Acceleration="local_auto"
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
        acceleration: Acceleration="local_auto"
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
        acceleration: Acceleration="local_auto"
    ) -> Message | Iterator[RawMessageStreamEvent]:
        """
        Create a message.

        Parameters:
            max_tokens (int): The maximum number of tokens to generate before stopping.
            messages (list): Input messages.
            model (str): Model predictor tag.
            stop_sequences (list): Custom text sequences that will cause the model to stop generating. Ignored unless the predictor natively supports it.
            stream (bool): Whether to incrementally stream the response.
            system (str | list): System prompt.
            temperature (float): Amount of randomness injected into the response.
            top_k (int): Only sample from the top K options for each subsequent token. Ignored unless the predictor natively supports it.
            top_p (float): Nucleus sampling coefficient.
            acceleration (Acceleration): Prediction acceleration.

        Returns:
            Message | Iterator[RawMessageStreamEvent]: Message or raw message stream events if streaming.
        """
        # Ensure we have a delegate
        if model not in self.__cache:
            self.__cache[model] = self.__create_delegate(model)
        # Make prediction
        delegate = self.__cache[model]
        result = delegate(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            stop_sequences=stop_sequences,
            stream=stream,
            system=system,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            acceleration=acceleration
        )
        # Return
        return result

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
        acceleration: Acceleration="local_auto"
    ) -> MessageStream:
        """
        Create a streaming message.

        Parameters:
            max_tokens (int): The maximum number of tokens to generate before stopping.
            messages (list): Input messages.
            model (str): Model predictor tag.
            stop_sequences (list): Custom text sequences that will cause the model to stop generating. Ignored unless the predictor natively supports it.
            system (str | list): System prompt.
            temperature (float): Amount of randomness injected into the response.
            top_k (int): Only sample from the top K options for each subsequent token. Ignored unless the predictor natively supports it.
            top_p (float): Nucleus sampling coefficient.
            acceleration (Acceleration): Prediction acceleration.

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
            acceleration=acceleration
        )
        return MessageStream(events)

    def __create_delegate(self, tag: str) -> MessageDelegate:
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        if not predictor:
            raise ValueError(
                f"{tag} cannot be used with Anthropic messages API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        # Check that there is only one required input parameter
        signature = predictor.signature
        required_inputs = [param for param in signature.inputs if not param.optional]
        if len(required_inputs) != 1:
            raise ValueError(
                f"{tag} cannot be used with Anthropic messages API because "
                "it has more than one required input parameter."
            )
        # Check that the input parameter is a message list
        _, input_param = get_parameter(required_inputs, dtype=Dtype.list)
        if input_param is None:
            raise ValueError(
                f"{tag} cannot be used with Anthropic messages API because "
                "it does not have a valid chat messages input parameter."
            )
        # Check whether the predictor is written against the OpenAI chat completions API.
        completion_param_idx = next((
            idx
            for idx, param in enumerate(signature.outputs)
            if (
                param.dtype == Dtype.dict and
                param.value_schema["title"] == "ChatCompletionChunk"
            )
        ), None)
        if completion_param_idx is not None:
            return self.__create_openai_delegate(
                signature,
                input_param,
                completion_param_idx
            )
        # Assume predictor was written against the Anthropic messages API.
        return self.__create_anthropic_delegate(
            tag,
            signature,
            input_param
        )

    def __create_openai_delegate(
        self,
        signature: Signature,
        input_param: Parameter,
        completion_param_idx: int
    ) -> MessageDelegate:
        # Get optional inputs
        _, max_output_tokens_param = get_parameter(
            signature.inputs,
            dtype=_INT_DTYPES,
            denotation="openai.chat.completions.max_output_tokens"
        )
        _, temperature_param = get_parameter(
            signature.inputs,
            dtype=_FLOAT_DTYPES,
            denotation="openai.chat.completions.temperature"
        )
        _, top_p_param = get_parameter(
            signature.inputs,
            dtype=_FLOAT_DTYPES,
            denotation="openai.chat.completions.top_p"
        )
        # Create delegate
        def delegate(
            *,
            max_tokens: int,
            messages: list[_MessageParamDict],
            model: str,
            stop_sequences: list[str] | None,
            stream: bool,
            system: str | list[_TextBlockParamDict] | None,
            temperature: float | None,
            top_k: int | None,
            top_p: float | None,
            acceleration: Acceleration
        ) -> Message | Iterator[RawMessageStreamEvent]:
            # Build prediction input map
            input_map: dict[str, object] = {
                input_param.name: _to_openai_messages(messages, system)
            }
            if max_output_tokens_param and max_tokens is not None:
                input_map[max_output_tokens_param.name] = max_tokens
            if temperature_param and temperature is not None:
                input_map[temperature_param.name] = temperature
            if top_p_param and top_p is not None:
                input_map[top_p_param.name] = top_p
            # Predict
            prediction_stream = self.__predictions.stream(
                tag=model,
                inputs=input_map,
                acceleration=acceleration
            )
            completion_stream = _gather_prediction_outputs(
                prediction_stream,
                completion_param_idx
            )
            chunks = map(_parse_completion_chunk, completion_stream)
            events = _stream_events(chunks)
            # Return
            if stream:
                return events
            else:
                return MessageStream(events).get_final_message()
        # Return
        return delegate

    def __create_anthropic_delegate(
        self,
        tag: str,
        signature: Signature,
        input_param: Parameter
    ) -> MessageDelegate:
        # Get optional inputs
        _, max_tokens_param = get_parameter(
            signature.inputs,
            dtype=_INT_DTYPES,
            denotation="openai.chat.completions.max_output_tokens"
        )
        _, stop_sequences_param = get_parameter(
            signature.inputs,
            dtype=Dtype.list,
            denotation="anthropic.messages.stop_sequences"
        )
        _, temperature_param = get_parameter(
            signature.inputs,
            dtype=_FLOAT_DTYPES,
            denotation="openai.chat.completions.temperature"
        )
        _, top_k_param = get_parameter(
            signature.inputs,
            dtype=_INT_DTYPES,
            denotation="anthropic.messages.top_k"
        )
        _, top_p_param = get_parameter(
            signature.inputs,
            dtype=_FLOAT_DTYPES,
            denotation="openai.chat.completions.top_p"
        )
        # Get message output param
        message_param_idx = next((
            idx
            for idx, param in enumerate(signature.outputs)
            if param.dtype == Dtype.dict
        ), None)
        if message_param_idx is None:
            raise ValueError(
                f"{tag} cannot be used with Anthropic messages API because "
                "it does not have a valid message output parameter."
            )
        # Create delegate
        def delegate(
            *,
            max_tokens: int,
            messages: list[_MessageParamDict],
            model: str,
            stop_sequences: list[str] | None,
            stream: bool,
            system: str | list[_TextBlockParamDict] | None,
            temperature: float | None,
            top_k: int | None,
            top_p: float | None,
            acceleration: Acceleration
        ) -> Message | Iterator[RawMessageStreamEvent]:
            # Build prediction input map, folding the system prompt into the messages.
            # Predictors that need the system prompt separately can filter it out.
            input_messages = (
                [{ "role": "system", "content": system }, *messages]
                if system is not None
                else messages
            )
            input_map: dict[str, object] = { input_param.name: input_messages }
            if max_tokens_param and max_tokens is not None:
                input_map[max_tokens_param.name] = max_tokens
            if stop_sequences_param and stop_sequences is not None:
                input_map[stop_sequences_param.name] = stop_sequences
            if temperature_param and temperature is not None:
                input_map[temperature_param.name] = temperature
            if top_k_param and top_k is not None:
                input_map[top_k_param.name] = top_k
            if top_p_param and top_p is not None:
                input_map[top_p_param.name] = top_p
            # Predict
            prediction_stream = self.__predictions.stream(
                tag=model,
                inputs=input_map,
                acceleration=acceleration
            )
            output_stream = _gather_prediction_outputs(
                prediction_stream,
                message_param_idx
            )
            events = _parse_message_events(output_stream)
            # Return
            if stream:
                return events
            else:
                return MessageStream(events).get_final_message()
        # Return
        return delegate

class MessageStream:
    """
    Message stream context manager.
    """

    def __init__(self, events: Iterator[RawMessageStreamEvent]):
        self.__events = events
        self.__message: Message | None = None

    def __enter__(self) -> MessageStream:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
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
                self.__message.content.append(event.content_block.model_copy())
            case RawContentBlockDeltaEvent() if self.__message is not None:
                block = self.__message.content[event.index]
                if isinstance(event.delta, TextDelta) and isinstance(block, TextBlock):
                    block.text += event.delta.text
                elif isinstance(event.delta, ThinkingDelta) and isinstance(block, ThinkingBlock):
                    block.thinking += event.delta.thinking
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

def _to_openai_messages(
    messages: list[_MessageParamDict],
    system: str | list[_TextBlockParamDict] | None
) -> list[_MessageDict]:
    result = list[_MessageDict]()
    if system is not None:
        result.append({ "role": "system", "content": _flatten_content(system) })
    result += [
        { "role": message["role"], "content": _flatten_content(message["content"]) }
        for message in messages
    ]
    return result

def _flatten_content(content: str | list[_TextBlockParamDict]) -> str:
    if isinstance(content, str):
        return content
    if any(block["type"] != "text" for block in content):
        raise ValueError(
            "Failed to create message because only `text` content "
            "blocks are currently supported"
        )
    return "".join(block["text"] for block in content)

def _gather_prediction_outputs(
    stream: Iterator[Prediction],
    output_param_idx: int
) -> Iterator[object]:
    for prediction in stream:
        if prediction.error:
            raise RuntimeError(prediction.error)
        yield prediction.results[output_param_idx]

def _parse_completion_chunk(data: object) -> ChatCompletionChunk:
    try:
        return TypeAdapter(ChatCompletionChunk).validate_python(data)
    except ValidationError:
        pass
    raise ValueError(
        f"Failed to parse chat completion chunk from model output: {data}. "
        "Chat predictors must yield `ChatCompletionChunk` outputs."
    )

def _stream_events(chunks: Iterator[ChatCompletionChunk]) -> Iterator[RawMessageStreamEvent]:
    started = False
    block_idx = -1
    block_type: Literal["thinking", "text"] | None = None
    stop_reason: StopReason = "end_turn"
    usage = Usage(output_tokens=0)
    for chunk in chunks:
        if not started:
            yield RawMessageStartEvent(message=Message(
                id=chunk.id,
                content=[],
                model=chunk.model,
                stop_reason=None,
                usage=Usage(input_tokens=0, output_tokens=0)
            ))
            started = True
        if chunk.usage:
            details = chunk.usage.prompt_tokens_details
            cache_read = details.cached_tokens if details else None
            cache_write = details.cache_write_tokens if details else None
            usage.input_tokens = chunk.usage.prompt_tokens - (cache_read or 0) - (cache_write or 0)
            usage.output_tokens += chunk.usage.completion_tokens
            usage.cache_read_input_tokens = cache_read
            usage.cache_creation_input_tokens = cache_write
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue
        if choice.finish_reason:
            stop_reason = _STOP_REASON_MAP.get(choice.finish_reason, "end_turn")
        if choice.delta is None:
            continue
        deltas: list[tuple[Literal["thinking", "text"], str | None]] = [
            ("thinking", choice.delta.reasoning_content),
            ("text", choice.delta.content)
        ]
        for kind, text in deltas:
            if not text:
                continue
            if block_type != kind:
                if block_type is not None:
                    yield RawContentBlockStopEvent(index=block_idx)
                block_idx += 1
                block_type = kind
                yield RawContentBlockStartEvent(
                    index=block_idx,
                    content_block=(
                        ThinkingBlock(thinking="")
                        if kind == "thinking"
                        else TextBlock(text="")
                    )
                )
            yield RawContentBlockDeltaEvent(
                index=block_idx,
                delta=(
                    ThinkingDelta(thinking=text)
                    if kind == "thinking"
                    else TextDelta(text=text)
                )
            )
    if block_type is not None:
        yield RawContentBlockStopEvent(index=block_idx)
    yield RawMessageDeltaEvent(
        delta=RawMessageDeltaEvent.MessageDelta(stop_reason=stop_reason),
        usage=usage
    )
    yield RawMessageStopEvent()

def _parse_message_events(outputs: Iterator[object]) -> Iterator[RawMessageStreamEvent]:
    for output in outputs:
        try:
            event = TypeAdapter(RawMessageStreamEvent).validate_python(output)
        except ValidationError:
            raise ValueError(
                f"Failed to parse message stream event from model output: {output}. "
                "Message predictors must yield `RawMessageStreamEvent` outputs."
            )
        yield event

_STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
    "tool_calls": "tool_use"
}

_INT_DTYPES = {
    Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
    Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64
}

_FLOAT_DTYPES = { Dtype.float32, Dtype.float64 }