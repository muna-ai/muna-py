# 
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable
from pydantic import TypeAdapter
from typing import overload, Iterator, Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype, Prediction
from ..remote import RemoteAcceleration
from ..remote.remote import RemotePredictionService
from .annotations import get_parameter
from .schema import ChatCompletion, ChatCompletionChunk, Message, _MessageDict, _ResponseFormatDict

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
        max_output_tokens: int | None=None,
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
        max_output_tokens: int | None=None,
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
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None=None,
        max_output_tokens: int | None=None,
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
            max_tokens (int): Maximum output tokens.
            acceleration (Acceleration | RemoteAcceleration): Prediction acceleration.
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
            max_output_tokens=max_output_tokens,
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
        completions_param_idx = next((
            idx
            for idx, param in enumerate(signature.outputs)
            if (
                param.type == Dtype.dict and
                param.value_schema["title"] in { "ChatCompletion", "ChatCompletionChunk" }
            )
        ), None)
        if completions_param_idx is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI chat completions API because "
                "it does not have a valid chat completion output parameter."
            )
        # Get usage output param
        usage_param_idx = next((
            idx
            for idx, param in enumerate(signature.outputs)
            if param.value_schema and param.value_schema.get("title") == "Usage"
        ), None)
        # Create delegate
        def delegate( # INCOMPLETE
            *,
            messages: list[Message | _MessageDict],
            model: str,
            stream: bool,
            response_format: _ResponseFormatDict | None,
            reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None,
            max_output_tokens: int | None,
            temperature: float | None,
            top_p: float | None,
            frequency_penalty: float | None,
            presence_penalty: float | None,
            acceleration: Acceleration | RemoteAcceleration
        ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
            pass
        # Return
        return delegate

    #     # Build prediction input dictionary
    #     adapter = TypeAdapter(list[Message])
    #     messages = adapter.validate_python(messages)
    #     inputs = {
    #         "messages": adapter.dump_python(messages, mode="json", by_alias=True),
    #         "stream": stream,
    #         "max_tokens": max_tokens
    #     }
    #     # Predict
    #     if stream:
    #         stream_prediction_func = (
    #             self.__remote_predictions.stream
    #             if acceleration.startswith("remote_")
    #             else self.__predictions.stream
    #         )
    #         prediction_stream = stream_prediction_func(
    #             tag=model,
    #             inputs=inputs,
    #             acceleration=acceleration
    #         )
    #         yield from map(self.__parse_response, prediction_stream)
    #     else:
    #         create_prediction_func = (
    #             self.__remote_predictions.create
    #             if acceleration.startswith("remote_")
    #             else self.__predictions.create
    #         )
    #         prediction = create_prediction_func(
    #             tag=model,
    #             inputs=inputs,
    #             acceleration=acceleration
    #         )
    #         return self.__parse_response(prediction)
    
    # def __parse_response(
    #     self,
    #     prediction: Prediction
    # ) -> ChatCompletion | ChatCompletionChunk:
    #     adapter = TypeAdapter(ChatCompletion | ChatCompletionChunk)
    #     completion_dict = prediction.results[0]
    #     completion = adapter.validate_python(completion_dict)
    #     return completion