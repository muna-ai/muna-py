#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from base64 import b64encode
from io import BytesIO
from os import PathLike
from pathlib import Path
from PIL import Image
from typing import BinaryIO, Literal

from ...types import Dtype, Parameter, Prediction, RemoteValue, Signature, Value
from .images import ImageResponse, OpenAIImageOptions, _image_response
from ._adapter import Adapter
from ._utils import get_parameter, validate_inputs, INTEGERS

ImageInput = Image.Image | bytes | PathLike[str] | str | BinaryIO | RemoteValue

class OpenAIImageEditAdapter(Adapter[Prediction, OpenAIImageOptions, ImageResponse]):
    """
    Convert Images edit inputs and outputs without executing predictions.
    """
    kind: Literal["openai.images.edit"] = "openai.images.edit"
    prompt: Parameter
    images: Parameter
    output: int
    count: Parameter | None = None
    width: Parameter | None = None
    height: Parameter | None = None

    @classmethod
    def from_signature(cls, signature: Signature) -> OpenAIImageEditAdapter:
        """
        Create an OpenAI image editing adapter from a predictor signature.
        """
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
        # Image output
        outputs = [
            idx
            for idx, param in enumerate(signature.outputs)
            if param.dtype == Dtype.image_list
        ]
        if len(outputs) != 1:
            raise ValueError("Requires one unambiguous image_list output")
        output = outputs[0]
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
        Create prediction inputs from an image editing request.
        """
        inputs = {self.images.name: image_list(images)}
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
        options: OpenAIImageOptions | None = None
    ) -> ImageResponse:
        """
        Adapt a prediction to an OpenAI image editing response.
        """
        return _image_response(prediction, self.output, options)

def remote_image(value: ImageInput) -> RemoteValue:
    if isinstance(value, RemoteValue):
        if value.dtype != Dtype.image:
            raise ValueError("Each image must have dtype image")
        return value
    if isinstance(value, Image.Image):
        buffer = BytesIO()
        value.save(buffer, format="PNG")
        data = buffer.getvalue()
    elif isinstance(value, str) and value.startswith(("https://", "http://", "data:")):
        return RemoteValue(dtype=Dtype.image, data=value)
    elif isinstance(value, (str, PathLike)):
        data = Path(value).read_bytes()
    elif isinstance(value, bytes):
        data = value
    elif hasattr(value, "read"):
        data = value.read()
    else:
        raise TypeError("Expected a Pillow image, encoded bytes, path, file, or RemoteValue")
    if not isinstance(data, bytes) or not data:
        raise ValueError("Encoded image data must be nonempty bytes")
    return RemoteValue(
        dtype=Dtype.image,
        data=f"data:image/*;base64,{b64encode(data).decode()}"
    )

def image_list(images: list[ImageInput]) -> list[Image.Image] | RemoteValue:
    if not isinstance(images, list) or not images:
        raise ValueError("images must be a nonempty list of images")
    # Keep already-decoded inputs native. Mixed/encoded lists remain wire values
    # until local execution explicitly requests decoding.
    if all(isinstance(image, Image.Image) for image in images):
        return list(images)
    return RemoteValue(
        dtype=Dtype.image_list,
        data=[remote_image(image) for image in images]
    )

