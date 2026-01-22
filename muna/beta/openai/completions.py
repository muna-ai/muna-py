# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections import defaultdict
from collections.abc import Callable
from pydantic import TypeAdapter, ValidationError
from typing import overload, Iterator, Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype, Prediction
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService
from .annotations import get_parameter
from .schema import (
    ChatCompletion, ChatCompletionChunk, ChatCompletionReasoningEffort,
    Choice, DeltaMessage, Message, _MessageDict, _ResponseFormatDict,
    StreamChoice
)

ChatCompletionDelegate = Callable[..., ChatCompletion | Iterator[ChatCompletionChunk]]

class ChatCompletionService:
    """
    Create chat completions.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService,
        remote_predictions: RemotePredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__remote_predictions = remote_predictions
        self.__cache = dict[str, ChatCompletionDelegate]()

    @overload
    def create(
        self,
        *,
        messages: list[Message | _MessageDict],
        model: str,
        stream: Literal[False]=False,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration | RemoteAcceleration="remote_auto"
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        messages: list[Message | _MessageDict],
        model: str,
        stream: Literal[True],
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration | RemoteAcceleration="remote_auto"
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        messages: list[Message | _MessageDict],
        model: str,
        stream: bool=False,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: ChatCompletionReasoningEffort | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration | RemoteAcceleration="remote_auto"
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """
        Create a chat completion.

        Parameters:
            messages (list): Messages for the conversation so far.
            model (str): Chat model predictor tag.
            stream (bool): Whether to stream responses.
            response_format (dict): Response format.
            reasoning_effort (ChatCompletionReasoningEffort): Reasoning effort for reasoning models.
            max_completion_tokens (int): Maximum completion tokens.
            temperature (float): Sampling temperature to use.
            top_p (float): Nucleus sampling coefficient.
            frequency_penalty (float): Token frequency penalty.
            presence_penalty (float): Token presence penalty.
            acceleration (Acceleration | RemoteAcceleration): Prediction acceleration.

        Returns:
            ChatCompletion | Iterator[ChatCompletionChunk]: Chat completion or chat completion chunks if streaming.
        """
        # Ensure we have a delegate
        if model not in self.__cache:
            self.__cache[model] = self.__create_delegate(model)
        # Make prediction
        delegate = self.__cache[model]
        result = delegate(
            messages=messages,
            model=model,
            stream=stream,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            acceleration=acceleration
        )
        # Return
        return result

    def __create_delegate(self, tag: str) -> ChatCompletionDelegate:
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        if not predictor:
            raise ValueError(
                f"{tag} cannot be used with OpenAI chat completions API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        # Check that there is only one required input parameter
        signature = predictor.signature
        required_inputs = [param for param in signature.inputs if not param.optional]
        if len(required_inputs) != 1:
            raise ValueError(
                f"{tag} cannot be used with OpenAI chat completions API because "
                "it has more than one required input parameter."
            )
        # Check that the input parameter is `list[Message]`
        _, input_param = get_parameter(required_inputs, dtype=Dtype.list)
        if input_param is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI chat completions API because "
                "it does not have a valid chat messages input parameter."
            )
        # Get optional inputs
        _, response_format_param = get_parameter(
            signature.inputs,
            dtype=Dtype.dict,
            denotation="openai.chat.completions.response_format"
        )
        _, reasoning_effort_param = get_parameter(
            signature.inputs,
            dtype=Dtype.string,
            denotation="openai.chat.completions.reasoning_effort"
        )
        _, max_output_tokens_param = get_parameter(
            signature.inputs,
            dtype={
                Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
                Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64
            },
            denotation="openai.chat.completions.max_output_tokens"
        )
        _, temperature_param = get_parameter(
            signature.inputs,
            dtype={ Dtype.float32, Dtype.float64 },
            denotation="openai.chat.completions.temperature"
        )
        _, top_p_param = get_parameter(
            signature.inputs,
            dtype={ Dtype.float32, Dtype.float64 },
            denotation="openai.chat.completions.top_p"
        )
        _, frequency_penalty_param = get_parameter(
            signature.inputs,
            dtype={ Dtype.float32, Dtype.float64 },
            denotation="openai.chat.completions.frequency_penalty"
        )
        _, presence_penalty_param = get_parameter(
            signature.inputs,
            dtype={ Dtype.float32, Dtype.float64 },
            denotation="openai.chat.completions.presence_penalty"
        )
        # Get chat completion output param
        completion_param_idx = next((
            idx
            for idx, param in enumerate(signature.outputs)
            if (
                param.dtype == Dtype.dict and
                param.value_schema["title"] in { "ChatCompletion", "ChatCompletionChunk" }
            )
        ), None)
        if completion_param_idx is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI chat completions API because "
                "it does not have a valid chat completion output parameter."
            )
        # Create delegate
        def delegate(
            *,
            messages: list[Message | _MessageDict],
            model: str,
            stream: bool,
            response_format: _ResponseFormatDict | None,
            reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None,
            max_completion_tokens: int | None,
            temperature: float | None,
            top_p: float | None,
            frequency_penalty: float | None,
            presence_penalty: float | None,
            acceleration: Acceleration | RemoteAcceleration
        ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
            # Get prediction creation and streaming functions (local or remote)
            stream_prediction_func = (
                self.__remote_predictions.stream
                if acceleration.startswith("remote_")
                else self.__predictions.stream
            )
            # Build prediction input map
            input_map = { input_param.name: messages }
            if response_format_param and response_format:
                input_map[response_format_param.name] = response_format
            if reasoning_effort_param and reasoning_effort:
                input_map[reasoning_effort_param.name] = reasoning_effort
            if max_output_tokens_param and max_completion_tokens is not None:
                input_map[max_output_tokens_param.name] = max_completion_tokens
            if temperature_param and temperature is not None:
                input_map[temperature_param.name] = temperature
            if top_p_param and top_p is not None:
                input_map[top_p_param.name] = top_p
            if frequency_penalty_param and frequency_penalty is not None:
                input_map[frequency_penalty_param.name] = frequency_penalty
            if presence_penalty_param and presence_penalty is not None:
                input_map[presence_penalty_param.name] = presence_penalty
            # Predict
            prediction_stream = stream_prediction_func(
                tag=model,
                inputs=input_map,
                acceleration=acceleration
            )
            completion_stream = _gather_completion_outputs(prediction_stream, completion_param_idx)
            # Return
            if stream:
                return map(_parse_chat_completion_chunk, completion_stream)
            else:
                return _parse_chat_completion(list(completion_stream))
        # Return
        return delegate

def _gather_completion_outputs(
    stream: Iterator[Prediction],
    completion_param_idx: int
) -> Iterator[object]:
    for prediction in stream:
        if prediction.error:
            raise RuntimeError(prediction.error)
        yield prediction.results[completion_param_idx]

def _parse_chat_completion(outputs: list[object]) -> ChatCompletion:
    if not outputs:
        raise ValueError(f"Failed to parse chat completion because model did not return any outputs")
    try:
        completions = TypeAdapter(list[ChatCompletion]).validate_python(outputs)
        return completions[-1]
    except ValidationError:
        pass
    try:
        chunks = TypeAdapter(list[ChatCompletionChunk]).validate_python(outputs)
        choices_map = defaultdict[int, list[StreamChoice]](list)
        for chunk in chunks:
            for choice in chunk.choices:
                choices_map[choice.index].append(choice)
        choices = [_create_chat_completion_choice(index, choices) for index, choices in choices_map.items()]
        chunk_usages = [chunk.usage for chunk in chunks if chunk.usage is not None]
        usage = ChatCompletion.Usage(
            prompt_tokens=sum(usage.prompt_tokens for usage in chunk_usages),
            completion_tokens=sum(usage.completion_tokens for usage in chunk_usages),
            total_tokens=sum(usage.total_tokens for usage in chunk_usages)
        )
        completion = ChatCompletion(
            id=chunks[0].id,
            created=chunks[0].created,
            model=chunks[0].model,
            choices=choices,
            usage=usage
        )
        return completion
    except ValidationError:
        pass
    raise ValueError(f"Failed to parse chat completion from model outputs: {outputs}")

def _parse_chat_completion_chunk(data: dict[str, object]) -> ChatCompletionChunk:
    try:
        return TypeAdapter(ChatCompletionChunk).validate_python(data)
    except ValidationError:
        pass
    try:
        completion = TypeAdapter(ChatCompletion).validate_python(data)
        chunk = ChatCompletionChunk(
            id=completion.id,
            created=completion.created,
            model=completion.model,
            choices=[StreamChoice(
                index=choice.index,
                delta=DeltaMessage(
                    role=choice.message.role,
                    content=choice.message.content
                ),
                finish_reason=choice.finish_reason
            ) for choice in completion.choices],
            usage=completion.usage,
        )
        return chunk
    except ValidationError:
        pass
    raise ValueError(f"Failed to parse streaming chat completion chunk from model output: {data}")

def _create_chat_completion_choice(
    index: int,
    choices: list[StreamChoice]
) -> Choice:
    role = choices[0].delta.role
    content = "".join(choice.delta.content for choice in choices if choice.delta)
    message = message=Message(role=role, content=content)
    finish_reason = next((choice.finish_reason for choice in choices if choice.finish_reason), None)
    result = Choice(
        index=index,
        message=message,
        finish_reason=finish_reason
    )
    return result