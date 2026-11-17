#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from base64 import b64encode
from io import BytesIO
from numpy import dtype, ndarray, save, savez_compressed
from pydantic import BaseModel, ConfigDict
from typing import Literal

from ...types import Dtype, Parameter, Prediction, RemoteValue, Signature, Value
from .edits import ImageInput, image_list
from .images import ImageData, ImageResponse
from ._adapter import Adapter
from ._utils import (
    created, decode, get_parameter, inline_base64,
    result, validate_inputs, INTEGERS
)

class OpenAIDepthOptions(BaseModel, **ConfigDict(extra="forbid", frozen=True, strict=True)):
    """
    Depth encoding options.
    """
    format: Literal["npy", "npz"] = "npz"

class OpenAIDepthAdapter(Adapter[Prediction, OpenAIDepthOptions, ImageResponse]):
    """
    Images edit inputs with lossless, per-map NPY/NPZ response items.
    """
    kind: Literal["openai.images.edit.depth"] = "openai.images.edit.depth"
    prompt: Parameter
    images: Parameter
    output: int
    count: Parameter | None = None
    width: Parameter | None = None
    height: Parameter | None = None

    @classmethod
    def from_signature(cls, signature: Signature) -> OpenAIDepthAdapter:
        signature = signature.model_copy(deep=True)
        # Prompt parameter
        _, prompt = get_parameter(
            signature.inputs,
            dtype=Dtype.string,
            denotation="openai.images.prompt"
        )
        if prompt is None:
            raise ValueError("Requires one scalar string input annotated with ImagePrompt")
        # Images parameter
        _, images = get_parameter(
            signature.inputs,
            dtype=Dtype.image_list,
            denotation="openai.chat.completions.images"
        )
        if images is None:
            raise ValueError("Requires one image_list input annotated with ChatImages")
        # Optional generation width
        _, width = get_parameter(
            signature.inputs,
            dtype=INTEGERS,
            denotation="openai.images.width"
        )
        if width is not None and not width.optional:
            raise ValueError("Image generation `width` input must be optional")
        # Optional generation height
        _, height = get_parameter(
            signature.inputs,
            dtype=INTEGERS,
            denotation="openai.images.height"
        )
        if height is not None and not height.optional:
            raise ValueError("Image generation `height` input must be optional")
        # Optional generation count
        _, count = get_parameter(
            signature.inputs,
            dtype=INTEGERS,
            denotation="openai.images.count"
        )
        if count is not None and not count.optional:
            raise ValueError("Image generation `count` input must be optional")
        # Validate inputs
        validate_inputs(signature, [prompt, images, width, height, count])
        # Depth output
        output, _ = get_parameter(
            signature.outputs,
            dtype=Dtype.array_list,
            denotation="depth_map"
        )
        if output is None:
            raise ValueError("Requires one array_list output annotated as depth_map")
        # Return
        return cls(
            prompt=prompt,
            images=images,
            output=output,
            count=count,
            width=width,
            height=height
        )

    def inputs(
        self,
        *,
        images: list[ImageInput],
        prompt: str | None = None,
        n: int | None = None,
        width: int | None = None,
        height: int | None = None
    ) -> dict[str, Value]:
        inputs = { self.images.name: image_list(images) }
        if prompt is not None:
            inputs[self.prompt.name] = prompt
        if self.count is not None and n is not None:
            inputs[self.count.name] = n
        if self.width is not None and width is not None:
            inputs[self.width.name] = width
        if self.height is not None and height is not None:
            inputs[self.height.name] = height
        return inputs

    def adapt(
        self,
        prediction: Prediction,
        options: OpenAIDepthOptions | None = None
    ) -> ImageResponse:
        options = options or OpenAIDepthOptions()
        maps = result(prediction, self.output)
        if isinstance(maps, RemoteValue):
            if maps.dtype != Dtype.array_list:
                raise ValueError("Expected an array_list depth output")
            maps = maps.data if isinstance(maps.data, list) else decode(maps, Dtype.array_list)
        if not isinstance(maps, list) or not maps:
            raise ValueError("Depth predictor must return a nonempty list of maps")
        return ImageResponse(
            data=[_depth_data(value, options) for value in maps],
            background=None,
            created=created(prediction),    
        )

def _depth_data(
    value: object,
    options: OpenAIDepthOptions
) -> ImageData:
    if isinstance(value, RemoteValue):
        if value.dtype not in (Dtype.float32, Dtype.float64):
            raise ValueError("Depth maps must have dtype float32 or float64")
        # Muna's tensor wire representation is NPZ; matching responses need no
        # intermediate ndarray. NPY requests go through the normal decoder.
        if options.format == "npz":
            payload = inline_base64(value, {"application/octet-stream", "application/vnd.muna.tensor"})
            if payload is not None:
                return ImageData(b64_json=payload, image=None)
        value = decode(value, value.dtype)
    if (
        not isinstance(value, ndarray) or
        value.ndim != 2 or
        min(value.shape, default=0) <= 0 or
        value.dtype not in (dtype("float32"), dtype("float64"))
    ):
        raise ValueError("Each depth map must be a nonempty (H,W) float32 or float64 array")
    buffer = BytesIO()
    if options.format == "npy":
        save(buffer, value, allow_pickle=False)
    else:
        savez_compressed(buffer, value)
    return ImageData(
        b64_json=b64encode(buffer.getvalue()).decode(),
        image=None
    )
