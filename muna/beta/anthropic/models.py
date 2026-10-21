# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from datetime import datetime, timezone

from ...services import PredictorService
from ...types import Parameter, Signature
from ..utils import rfc3339_to_unix, split_tag
from .schema import (
    CapabilitySupport, EffortCapability, ModelCapabilities, ModelInfo,
    ThinkingCapability, ThinkingTypes
)

class ModelService:
    """
    Describe models.
    """

    def __init__(self, predictors: PredictorService):
        self.__predictors = predictors

    def retrieve(self, model: str) -> ModelInfo:
        """
        Retrieve a model.

        Parameters:
            model (str): Predictor tag.
        """
        predictor = self.__predictors.retrieve(model)
        if not predictor:
            raise ValueError(
                f"{model} cannot be retrieved with the Anthropic models API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        return _create_model_info(
            predictor.tag,
            predictor.signature,
            created=rfc3339_to_unix(predictor.created)
        )

def _create_model_info(
    tag: str,
    signature: Signature,
    *,
    created: int | None
) -> ModelInfo:
    """
    Build the model info for a predictor from its signature. The limits and
    capabilities come from the signature's chat denotations, so what is
    advertised is exactly what the predictor accepts.

    `created` is the Unix time the model became available; Anthropic allows
    an epoch value when unknown, so `None` renders as `1970-01-01T00:00:00Z`.
    """
    _, name = split_tag(tag)
    inputs = signature.inputs
    thinking = _chat_effort_accepts(inputs, "low") is not None
    accepts = lambda level: CapabilitySupport(supported=bool(_chat_effort_accepts(inputs, level)))
    unsupported = CapabilitySupport(supported=False)
    return ModelInfo(
        id=tag,
        display_name=name,
        created_at=_unix_to_rfc3339(created or 0),
        max_input_tokens=_chat_context_length(inputs),
        max_tokens=_chat_max_output_tokens(inputs),
        capabilities=ModelCapabilities(
            batch=unsupported,
            citations=unsupported,
            code_execution=unsupported,
            context_management=unsupported,
            effort=EffortCapability(
                supported=thinking,
                low=accepts("low"),
                medium=accepts("medium"),
                high=accepts("high"),
                xhigh=accepts("xhigh"),
                max=accepts("max")
            ),
            image_input=CapabilitySupport(supported=_find_denoted(inputs, _CHAT_IMAGES) is not None),
            pdf_input=unsupported,
            structured_outputs=CapabilitySupport(supported=_find_denoted(inputs, _CHAT_RESPONSE_FORMAT) is not None),
            thinking=ThinkingCapability(
                supported=thinking,
                types=ThinkingTypes(
                    adaptive=unsupported,
                    enabled=CapabilitySupport(supported=thinking)
                )
            )
        )
    )

def _find_denoted(
    parameters: list[Parameter],
    denotation: str
) -> Parameter | None:
    """
    The first parameter carrying `denotation`, if any.
    """
    return next((p for p in parameters if p.denotation == denotation), None)

def _chat_context_length(inputs: list[Parameter]) -> int | None:
    """
    The model's total context window in tokens, from the chat messages input.
    """
    parameter = _find_denoted(inputs, _CHAT_MESSAGES)
    return parameter.context_length if parameter else None

def _chat_max_output_tokens(inputs: list[Parameter]) -> int | None:
    """
    The ceiling on output tokens, from the `max_output_tokens` input's range.
    """
    parameter = _find_denoted(inputs, _CHAT_MAX_OUTPUT_TOKENS)
    if parameter is None or parameter.max is None or parameter.max < 1:
        return None
    return int(min(parameter.max, _U32_MAX))

def _chat_effort_accepts(
    inputs: list[Parameter],
    level: str
) -> bool | None:
    """
    Whether the reasoning effort selector accepts `level`. `None` when the
    model has no selector; `True` for every level when the selector is free
    text (no enumeration).
    """
    parameter = _find_denoted(inputs, _CHAT_REASONING_EFFORT)
    if parameter is None:
        return None
    if parameter.enumeration is None:
        return True
    return any(member.value == level for member in parameter.enumeration)

def _unix_to_rfc3339(unix: int) -> str:
    """
    Unix seconds to `YYYY-MM-DDTHH:MM:SSZ`.
    """
    return datetime.fromtimestamp(unix, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

_CHAT_MESSAGES = "openai.chat.completions.messages"
_CHAT_MAX_OUTPUT_TOKENS = "openai.chat.completions.max_output_tokens"
_CHAT_REASONING_EFFORT = "openai.chat.completions.reasoning_effort"
_CHAT_IMAGES = "openai.chat.completions.images"
_CHAT_RESPONSE_FORMAT = "openai.chat.completions.response_format"
_U32_MAX = 2**32 - 1
