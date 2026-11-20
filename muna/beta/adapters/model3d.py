#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from base64 import b64decode, b64encode
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ...types import Dtype, Parameter, Prediction, RemoteValue, Signature, Value
from .edits import ImageInput, image_list
from .images import ImageData, ImageResponse
from ._adapter import Adapter
from ._utils import (
    created, decode, get_parameter, inline_base64,
    result, validate_inputs, INTEGERS
)

class OpenAIModel3DOptions(BaseModel, **ConfigDict(extra="forbid", frozen=True, strict=True)):
    """
    3D asset response options; the predictor documents its file format.
    """
    format: Literal["ply"] = "ply"

class OpenAIModel3DAdapter(Adapter[Prediction, OpenAIModel3DOptions, ImageResponse]):
    """
    Image-conditioned 3D assets returned as existing predictor-produced bytes.
    """
    kind: Literal["openai.images.edit.model3d"] = Field("openai.images.edit.model3d", init=False)
    prompt: Parameter
    images: Parameter
    output: int
    count: Parameter | None = None
    width: Parameter | None = None
    height: Parameter | None = None

    @classmethod
    def from_signature(cls, signature: Signature) -> OpenAIModel3DAdapter | None:
        """
        Create an OpenAI-compatible 3D model adapter from a predictor signature.
        """
        signature = signature.model_copy(deep=True)
        # Prompt parameter
        _, prompt = get_parameter(
            signature.inputs,
            dtype=Dtype.string,
            denotation="openai.images.prompt"
        )
        if prompt is None:
            return None
        # Images parameter
        _, images = get_parameter(
            signature.inputs,
            dtype=Dtype.image_list,
            denotation="openai.chat.completions.images"
        )
        if images is None:
            return None
        # Optional generation width
        _, width = get_parameter(
            signature.inputs,
            dtype=INTEGERS,
            denotation="openai.images.width"
        )
        if width is not None and not width.optional:
            return None
        # Optional generation height
        _, height = get_parameter(
            signature.inputs,
            dtype=INTEGERS,
            denotation="openai.images.height"
        )
        if height is not None and not height.optional:
            return None
        # Optional generation count
        _, count = get_parameter(
            signature.inputs,
            dtype=INTEGERS,
            denotation="openai.images.count"
        )
        if count is not None and not count.optional:
            return None
        # Validate inputs
        if not validate_inputs(signature, [prompt, images, width, height, count]):
            return None
        # 3D asset output
        output, _ = get_parameter(
            signature.outputs,
            dtype=Dtype.binary,
            denotation="3d"
        )
        if output is None:
            return None
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
        """
        Create prediction inputs from a 3D generation request.
        """
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
        options: OpenAIModel3DOptions | None = None
    ) -> ImageResponse:
        """
        Wrap the predictor's PLY bytes without converting its 3D representation.
        """
        options = options or OpenAIModel3DOptions()
        value = result(prediction, self.output)
        payload = None
        if isinstance(value, RemoteValue):
            if value.dtype != Dtype.binary:
                raise ValueError("Expected a binary 3D output")
            payload = inline_base64(value, {
                "application/octet-stream",
                "application/ply",
                "model/ply"
            })
        if payload is not None:
            # Only inspect the file magic; the potentially large body stays encoded.
            try:
                prefix = b64decode(payload[:8], validate=True)
            except ValueError as error:
                raise ValueError("Expected a PLY output") from error
        else:
            value = decode(value, Dtype.binary)
            if not isinstance(value, bytes):
                raise ValueError("Expected a binary 3D output")
            prefix = value[:6]
            payload = b64encode(value).decode()
        if not prefix.startswith((b"ply\n", b"ply\r\n")):
            raise ValueError("Expected a PLY output; the adapter does not convert 3D formats")
        return ImageResponse(
            data=[ImageData(b64_json=payload, image=None)],
            background=None,
            created=created(prediction),
        )
