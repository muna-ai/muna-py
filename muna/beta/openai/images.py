# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from base64 import b64encode
from collections.abc import Callable
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import BinaryIO
from PIL import Image
from time import time
from typing import Literal

from ...c import Value
from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device, Dtype
from ..annotations import get_parameter
from .schema import ImageData, ImageResponse

ImageSize = Literal[
    "auto",
    "1024x1024",
    "1536x1024",
    "1024x1536",
    "256x256",
    "512x512",
    "1792x1024",
    "1024x1792"
]

ImageInput = Image.Image | bytes | PathLike[str] | str | BinaryIO

ImageDelegate = Callable[..., ImageResponse]

class ImageService:
    """
    Image service.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache = dict[str, ImageDelegate]()
        self.__edit_cache = dict[str, ImageDelegate]()

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        background: Literal["auto", "transparent", "opaque"] | None = None,
        n: int | None = None,
        output_format: Literal["png", "jpeg", "webp", "raw"] | None = None,
        output_compression: int | None = None,
        size: ImageSize | None = None,
        acceleration: Acceleration = "local_auto",
        devices: list[Device | str] | None = None
    ) -> ImageResponse:
        """
        Generate an image given a prompt.

        Parameter:
            prompt (str): Text description of the desired image.
            model (str): Image generation model tag.
            background (str): Set transparency for the background of the generated image(s).
            n (int): Number of images to generate. 
            output_compression (int): The compression level for the generated images in range [0,100].
            output_format (str): The format in which the generated images are returned.
            size (str): The size of the generated images.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.

        Returns:
            ImageRepsonse: Generated image.
        """
        # Ensure we have a delegate
        if model not in self.__cache:
            self.__cache[model] = self.__create_delegate(model)
        # Make prediction
        delegate = self.__cache[model]
        result = delegate(
            prompt=prompt,
            model=model,
            background=background,
            n=n,
            output_compression=output_compression,
            output_format=output_format,
            size=size,
            acceleration=acceleration,
            devices=devices
        )
        # Return
        return result

    def edit(
        self,
        *,
        image: ImageInput | list[ImageInput],
        prompt: str | None = None,
        model: str,
        background: Literal["auto", "transparent", "opaque"] | None = None,
        n: int | None = None,
        output_format: Literal["png", "jpeg", "webp", "raw"] | None = None,
        output_compression: int | None = None,
        size: ImageSize | None = None,
        acceleration: Acceleration = "local_auto",
        devices: list[Device | str] | None = None
    ) -> ImageResponse:
        """
        Edit an image with a given prompt.

        Parameters:
            image (ImageInput | list): Input image(s), always bound as an image list.
            prompt (str): Editing prompt. May be omitted if the predictor declares it optional.
            background (str): Background mode.
            n (int): Result image count.
            output_format (str): Result image format.
            output_compression (int): Result image compression.
            size (str): The size of the generated images.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.

        Returns:
            ImageResponse: Editing result.
        """
        values = image if isinstance(image, list) else [image]
        if not values:
            raise ValueError("images.edit requires at least one input image")
        images = [_load_image(value) for value in values]
        if model not in self.__edit_cache:
            self.__edit_cache[model] = self.__create_delegate(model, edit=True)
        return self.__edit_cache[model](
            image=images,
            prompt=prompt,
            model=model,
            background=background,
            n=n,
            output_format=output_format,
            output_compression=output_compression,
            size=size,
            acceleration=acceleration,
            devices=devices
        )

    def __create_delegate(self, tag: str, *, edit: bool=False) -> ImageDelegate:
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        if not predictor:
            raise ValueError(
                f"{tag} cannot be used with OpenAI image API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        signature = predictor.signature
        required_inputs = [param for param in signature.inputs if not param.optional]
        image_param = None
        if edit:
            prompts = [p for p in signature.inputs if p.denotation == "openai.images.prompt"]
            if len(prompts) != 1 or prompts[0].dtype != Dtype.string:
                raise ValueError(
                    f"{tag} requires one scalar string input annotated with "
                    "Annotations.ImagePrompt for OpenAI Images edit compatibility"
                )
            candidates = [p for p in signature.inputs if p.dtype == Dtype.image_list]
            annotated = [p for p in candidates if p.denotation == "openai.chat.completions.images"]
            if len(annotated) == 1:
                image_param = annotated[0]
            elif not annotated and len(candidates) == 1:
                image_param = candidates[0]
            else:
                raise ValueError(f"{tag} requires one unambiguous image_list input for OpenAI Images edit compatibility")
        else:
            prompts = [p for p in required_inputs if p.dtype == Dtype.string]
            if len(prompts) != 1:
                raise ValueError(f"{tag} does not have a valid prompt input for this OpenAI image API")
        prompt_param = prompts[0]
        supported_inputs = { prompt_param.name }
        if image_param is not None:
            supported_inputs.add(image_param.name)
        if any(param.name not in supported_inputs for param in required_inputs):
            raise ValueError(f"{tag} has unsupported required inputs for this OpenAI image API")
        # Get the generation width parameter (optional)
        _, width_param = get_parameter(
            signature.inputs,
            dtype=_NUMERIC_DTYPES,
            denotation="openai.images.width"
        )
        # Get the generation height parameter (optional)
        _, height_param = get_parameter(
            signature.inputs,
            dtype=_NUMERIC_DTYPES,
            denotation="openai.images.height"
        )
        # Get the generation count parameter (optional)
        _, count_param = get_parameter(
            signature.inputs,
            dtype=_NUMERIC_DTYPES,
            denotation="openai.images.count"
        )
        # Get the image output parameter index
        image_param_idx, _ = get_parameter(signature.outputs, dtype=Dtype.image_list)
        if image_param_idx is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI image API because "
                "it has no image outputs."
            )
        # Define delegate
        def delegate(
            *,
            prompt: str | None,
            image: list[Image.Image] | None=None,
            model: str,
            background: Literal["auto", "transparent", "opaque"] | None,
            n: int | None,
            output_format: Literal["png", "jpeg", "webp", "raw"] | None,
            output_compression: int | None,
            size: ImageSize | None,
            acceleration: Acceleration,
            devices: list[Device | str] | None,
        ) -> ImageResponse:
            if prompt is None:
                if not prompt_param.optional:
                    raise ValueError("Images API prompt is required by this predictor")
            elif not isinstance(prompt, str):
                raise TypeError("Images API prompt must be a scalar string")
            if n is not None and n < 1:
                raise ValueError("Image count must be positive")
            # Build prediction input map
            requested_width, requested_height = _get_image_size(size)
            input_map = { }
            if prompt is not None:
                input_map[prompt_param.name] = prompt
            if image_param is not None:
                input_map[image_param.name] = image
            if n is not None and count_param is not None:
                input_map[count_param.name] = n
            if requested_width is not None and width_param is not None:
                input_map[width_param.name] = requested_width
            if requested_height is not None and height_param is not None:
                input_map[height_param.name] = requested_height
            # Create prediction
            prediction = self.__predictions.create(
                tag=model,
                inputs=input_map,
                acceleration=acceleration,
                devices=devices
            )
            # Check for error
            if prediction.error:
                raise RuntimeError(prediction.error)            
            # Create response
            images: list[Image.Image] = prediction.results[image_param_idx]
            image_data = [_create_image_data(
                image,
                output_format=output_format,
                output_compression=output_compression
            ) for image in images]
            response = ImageResponse(
                data=image_data,
                background="opaque",
                created=int(time()),
                usage=None
            )
            # Return
            return response
        # Return
        return delegate

def _load_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            return image.copy()
    if isinstance(value, (str, PathLike)):
        with Image.open(Path(value)) as image:
            return image.copy()
    # Detach file-backed images so callers can close their input streams.
    with Image.open(value) as image:
        return image.copy()

def _create_image_data(
    image: Image.Image,
    *,
    output_format: Literal["png", "jpeg", "webp", "raw"] | None,
    output_compression: int | None,
) -> ImageData:
    match output_format or "png":
        case "raw":     return ImageData(b64_json=None, image=image)
        case "webp":    raise ValueError(f"webp output format is not yet supported")
        case _:
            mime = f"image/{output_format or 'png'}"
            if output_compression is not None:
                mime += f";quality={output_compression}"
            with Value.from_object(image) as image_value:
                data = image_value.serialize(mime)
            return ImageData(b64_json=b64encode(data).decode("ascii"), image=None)

def _get_image_size(size: ImageSize | None) -> tuple[int | None, int | None]:
    try:
        w, h = size.split("x", 1)
        return int(w), int(h)
    except (ValueError, AttributeError):
        return None, None

_NUMERIC_DTYPES = {
    Dtype.int8, Dtype.int16, Dtype.int32, Dtype.int64,
    Dtype.uint8, Dtype.uint16, Dtype.uint32, Dtype.uint64
}