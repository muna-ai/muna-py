# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from base64 import b64decode
from collections import defaultdict
from collections.abc import Callable
from io import BytesIO
from numpy import ndarray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from requests import get as request_get
from typing import overload, Annotated, Iterator, Literal

from ...c import Value
from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype, Parameter, Prediction
from ..annotations import get_parameter
from .schema import (
    ChatCompletion, ChatCompletionChunk, ChatCompletionContentPartFile,
    ChatCompletionContentPartImage, ChatCompletionContentPartInputAudio,
    ChatCompletionContentPartRefusal, ChatCompletionContentPartText,
    ChatCompletionFunctionTool, ChatCompletionMessageFunctionToolCall,
    ChatCompletionReasoningEffort, ChatCompletionToolChoice, Choice,
    Message, _MessageDict, _ResponseFormatDict, StreamChoice
)

ChatCompletionDelegate = Callable[..., ChatCompletion | Iterator[ChatCompletionChunk]]

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
        self.__cache = dict[str, ChatCompletionDelegate]()

    @overload
    def create(
        self,
        *,
        messages: list[Message | _MessageDict],
        model: str,
        stream: Literal[False]=False,
        tools: list[ChatCompletionFunctionTool | dict] | None=None,
        tool_choice: ChatCompletionToolChoice | None=None,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration="local_auto"
    ) -> ChatCompletion: ...

    @overload
    def create(
        self,
        *,
        messages: list[Message | _MessageDict],
        model: str,
        stream: Literal[True],
        tools: list[ChatCompletionFunctionTool | dict] | None=None,
        tool_choice: ChatCompletionToolChoice | None=None,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration="local_auto"
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self,
        *,
        messages: list[Message | _MessageDict],
        model: str,
        stream: bool=False,
        tools: list[ChatCompletionFunctionTool | dict] | None=None,
        tool_choice: ChatCompletionToolChoice | None=None,
        response_format: _ResponseFormatDict | None=None,
        reasoning_effort: ChatCompletionReasoningEffort | None=None,
        max_completion_tokens: int | None=None,
        temperature: float | None=None,
        top_p: float | None=None,
        frequency_penalty: float | None=None,
        presence_penalty: float | None=None,
        acceleration: Acceleration="local_auto"
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
            frequency_penalty (float): Token frequency penalty.
            presence_penalty (float): Token presence penalty.
            acceleration (Acceleration): Prediction acceleration.

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
            tools=tools,
            tool_choice=tool_choice,
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
        _, images_param = get_parameter(
            signature.inputs,
            dtype=Dtype.image_list,
            denotation="openai.chat.completions.images"
        )
        _, audios_param = get_parameter(
            signature.inputs,
            dtype=Dtype.array_list,
            denotation="openai.chat.completions.audios"
        )
        _, tools_param = get_parameter(
            signature.inputs,
            dtype=Dtype.list,
            denotation="openai.chat.completions.tools"
        )
        # Get chat completion chunk output param
        completion_param_idx = next((
            idx
            for idx, param in enumerate(signature.outputs)
            if (
                param.dtype == Dtype.dict and
                param.value_schema["title"] == "ChatCompletionChunk"
            )
        ), None)
        if completion_param_idx is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI chat completions API because "
                "it does not have a valid chat completion chunk output parameter. "
                "Chat predictors must yield `ChatCompletionChunk` outputs."
            )
        # Create delegate
        def delegate(
            *,
            messages: list[Message | _MessageDict],
            model: str,
            stream: bool,
            tools: list[ChatCompletionFunctionTool | dict] | None,
            tool_choice: ChatCompletionToolChoice | None,
            response_format: _ResponseFormatDict | None,
            reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] | None,
            max_completion_tokens: int | None,
            temperature: float | None,
            top_p: float | None,
            frequency_penalty: float | None,
            presence_penalty: float | None,
            acceleration: Acceleration
        ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
            # Check tool support
            if tools and tools_param is None:
                raise ValueError(
                    f"{model} does not support tool calling because it does "
                    "not declare a tools input parameter."
                )
            # Build prediction input map
            conversation = _normalize_conversation(
                messages,
                images_param=images_param,
                audios_param=audios_param
            )
            input_map = { input_param.name: [
                message.model_dump(exclude_none=True)
                for message in conversation.messages
            ] }
            if images_param and conversation.images:
                input_map[images_param.name] = conversation.images
            if audios_param and conversation.audios:
                input_map[audios_param.name] = conversation.audios
            if tools and tool_choice != "none":
                input_map[tools_param.name] = [
                    tool.model_dump(exclude_none=True)
                    if isinstance(tool, BaseModel)
                    else tool
                    for tool in tools
                ]
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
            prediction_stream = self.__predictions.stream(
                tag=model,
                inputs=input_map,
                acceleration=acceleration
            )
            completion_stream = _gather_completion_outputs(prediction_stream, completion_param_idx)
            chunks = map(_parse_chat_completion_chunk, completion_stream)
            # Return
            if stream:
                return chunks
            else:
                return _merge_chunks(list(chunks))
        # Return
        return delegate

def _normalize_conversation(
    messages: list[Message | _MessageDict],
    *,
    images_param: Parameter | None,
    audios_param: Parameter | None
) -> _NormalizedConversation:
    """
    Normalize content parts: flatten text parts, decode media parts into
    parallel lists correlated by order of appearance across all messages.
    """
    normalized: list[_NormalizedMessage] = []
    images: list[Image.Image] = []
    audios: list[ndarray] = []
    for raw in messages:
        message = (
            TypeAdapter(Message).validate_python(raw)
            if isinstance(raw, dict)
            else raw
        )
        if not isinstance(message.content, list):
            normalized.append(_NormalizedMessage(**message.model_dump()))
            continue
        parts: list[_NormalizedPart] = []
        for part in message.content:
            match part:
                case ChatCompletionContentPartText():
                    parts.append(part)
                case ChatCompletionContentPartRefusal():
                    # Replayed assistant refusals flatten as text.
                    parts.append(ChatCompletionContentPartText(text=part.refusal))
                case ChatCompletionContentPartImage() if images_param:
                    images.append(_decode_image(part.image_url.url))
                    parts.append(_ImagePlaceholder())
                case ChatCompletionContentPartInputAudio() if audios_param:
                    audios.append(_decode_audio(
                        part.input_audio,
                        sample_rate=audios_param.sample_rate
                    ))
                    parts.append(_AudioPlaceholder())
                case ChatCompletionContentPartImage() | ChatCompletionContentPartInputAudio():
                    raise ValueError(f"`{part.type}` content is not supported by this model.")
                case ChatCompletionContentPartFile():
                    raise ValueError("File content parts are not yet supported.")
        # Text-only parts flatten to a plain string; mixed parts stay an array.
        content: str | list[_NormalizedPart]
        if all(isinstance(part, ChatCompletionContentPartText) for part in parts):
            content = "\n".join(part.text for part in parts)
        else:
            content = parts
        normalized.append(_NormalizedMessage(
            role=message.role,
            content=content,
            reasoning_content=message.reasoning_content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id
        ))
    return _NormalizedConversation(
        messages=normalized,
        images=images,
        audios=audios
    )

def _decode_image(url: str) -> Image.Image:
    """
    Decode an image content part URL (base64 data URL or remote URL) into
    an RGBA pixel buffer.
    """
    if url.startswith("data:"):
        _, _, encoded = url.partition(",")
        data = b64decode(encoded)
    else:
        response = request_get(url, timeout=10, stream=True)
        response.raise_for_status()
        data = response.raw.read(_MAX_IMAGE_FETCH_BYTES + 1, decode_content=True)
        if len(data) > _MAX_IMAGE_FETCH_BYTES:
            raise ValueError(f"Image at {url} exceeds the maximum size of {_MAX_IMAGE_FETCH_BYTES} bytes.")
    image = Image.open(BytesIO(data))
    return image.convert("RGBA")

def _decode_audio(
    input_audio: ChatCompletionContentPartInputAudio.InputAudio,
    *,
    sample_rate: int
) -> ndarray:
    """
    Decode an audio content part into linear PCM samples at the
    predictor's declared sample rate.
    """
    data = b64decode(input_audio.data)
    with Value.from_bytes(data, f"audio/{input_audio.format};rate={sample_rate}") as value:
        return value.to_object()

def _gather_completion_outputs(
    stream: Iterator[Prediction],
    completion_param_idx: int
) -> Iterator[object]:
    for prediction in stream:
        if prediction.error:
            raise RuntimeError(prediction.error)
        yield prediction.results[completion_param_idx]

def _merge_chunks(chunks: list[ChatCompletionChunk]) -> ChatCompletion:
    if not chunks:
        raise ValueError(f"Failed to parse chat completion because model did not return any outputs")
    choices_map = defaultdict[int, list[StreamChoice]](list)
    for chunk in chunks:
        for choice in chunk.choices:
            choices_map[choice.index].append(choice)
    choices = [_create_chat_completion_choice(index, choices) for index, choices in choices_map.items()]
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

def _parse_chat_completion_chunk(data: dict[str, object]) -> ChatCompletionChunk:
    try:
        return TypeAdapter(ChatCompletionChunk).validate_python(data)
    except ValidationError:
        pass
    raise ValueError(
        f"Failed to parse chat completion chunk from model output: {data}. "
        "Chat predictors must yield `ChatCompletionChunk` outputs."
    )

def _create_chat_completion_choice(
    index: int,
    choices: list[StreamChoice]
) -> Choice:
    role = choices[0].delta.role
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
    message = Message(
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

class _ImagePlaceholder(BaseModel):
    """
    HF-canonical placeholder: the nth image part across the conversation
    corresponds to the nth entry of the `images` input.
    """
    type: Literal["image"] = Field("image", init=False)

class _AudioPlaceholder(BaseModel):
    """
    HF-canonical placeholder: the nth audio part across the conversation
    corresponds to the nth entry of the `audios` input.
    """
    type: Literal["audio"] = Field("audio", init=False)

_NormalizedPart = Annotated[
    ChatCompletionContentPartText   |
    _ImagePlaceholder               |
    _AudioPlaceholder,
    Field(discriminator="type")
]

class _NormalizedMessage(BaseModel):
    """
    Chat message after content-part translation: text-only content is a
    plain string; multimodal content is a part list with media replaced
    by placeholders that index into the parallel media inputs. Tool fields
    pass through untouched so agent turns replay into the chat template.
    """
    role: Literal["assistant", "user", "system", "tool"]
    content: str | list[_NormalizedPart] | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatCompletionMessageFunctionToolCall] | None = None
    tool_call_id: str | None = None

class _NormalizedConversation(BaseModel, **ConfigDict(arbitrary_types_allowed=True)):
    """
    Result of content-part translation: normalized messages plus the
    parallel media lists their placeholders index into.
    """
    messages: list[_NormalizedMessage]
    images: list[Image.Image]
    audios: list[ndarray]

_MAX_IMAGE_FETCH_BYTES = 20 * 1024 * 1024