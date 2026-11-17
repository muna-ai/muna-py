# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

from ..adapters.messages import (
    StopReason,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ContentBlock,
    Message,
    Usage,
    TextDelta,
    ThinkingDelta,
    InputJSONDelta,
    RawMessageStartEvent,
    RawContentBlockStartEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStopEvent,
    RawMessageStreamEvent,
    _TextBlockParamDict,
    _MessageParamDict,
    _ImageBlockParamDict,
    _Base64ImageSourceDict,
    _URLImageSourceDict,
    _ToolUseBlockParamDict,
    _ToolResultBlockParamDict,
    _ThinkingBlockParamDict,
    ToolParam
)

class CapabilitySupport(BaseModel):
    """
    Whether a model supports a capability.
    """
    supported: bool = Field(description="Whether this capability is supported by the model.")

class EffortCapability(BaseModel):
    """
    Reasoning effort levels a model accepts.
    """
    supported: bool = Field(description="Whether the model supports an effort selector at all.")
    low: CapabilitySupport = Field(description="Whether the model supports the `low` effort level.")
    medium: CapabilitySupport = Field(description="Whether the model supports the `medium` effort level.")
    high: CapabilitySupport = Field(description="Whether the model supports the `high` effort level.")
    xhigh: CapabilitySupport = Field(description="Whether the model supports the `xhigh` effort level.")
    max: CapabilitySupport = Field(description="Whether the model supports the `max` effort level.")

class ThinkingTypes(BaseModel):
    """
    Thinking type configurations a model accepts.
    """
    adaptive: CapabilitySupport = Field(description="Whether the model supports thinking with type `adaptive`.")
    enabled: CapabilitySupport = Field(description="Whether the model supports thinking with type `enabled`.")

class ThinkingCapability(BaseModel):
    """
    Thinking capability of a model.
    """
    supported: bool = Field(description="Whether the model can think before answering.")
    types: ThinkingTypes = Field(description="Supported thinking type configurations.")

class ModelCapabilities(BaseModel):
    """
    Model capabilities.
    """
    batch: CapabilitySupport = Field(description="Whether the model supports the Batch API.")
    citations: CapabilitySupport = Field(description="Whether the model supports citation generation.")
    code_execution: CapabilitySupport = Field(description="Whether the model supports code execution tools.")
    context_management: CapabilitySupport = Field(description="Whether the model supports server-side context management.")
    effort: EffortCapability = Field(description="Reasoning effort support and available levels.")
    image_input: CapabilitySupport = Field(description="Whether the model accepts image content blocks.")
    pdf_input: CapabilitySupport = Field(description="Whether the model accepts PDF content blocks.")
    structured_outputs: CapabilitySupport = Field(description="Whether the model supports structured outputs.")
    thinking: ThinkingCapability = Field(description="Thinking capability and supported type configurations.")

class ModelInfo(BaseModel):
    """
    Model information.
    """
    id: str = Field(description="Model identifier (the predictor tag).")
    type: Literal["model"] = Field(default="model", init=False)
    display_name: str = Field(description="Human-readable name, the tag's `name`.")
    created_at: str = Field(description="Release time as an RFC 3339 timestamp; the epoch when unknown.")
    max_input_tokens: int | None = Field(description="Maximum input context window in tokens.")
    max_tokens: int | None = Field(description="Maximum value of the `max_tokens` parameter.")
    capabilities: ModelCapabilities = Field(description="Model capabilities.")
