# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from base64 import b64encode
from collections.abc import Callable
from PIL import Image
from time import time
from typing import Literal

from ...c import Value
from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
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

ImageDelegate = Callable[..., ImageResponse]

class ImageService: # DEPLOY
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

    def create(
        self,
        *,
        prompt: str,
        model: str,
        background: Literal["auto", "transparent", "opaque"] | None=None,
        n: int | None=None,
        output_format: Literal["png", "jpeg", "webp", "raw"]=None,
        output_compression: int | None=None,
        size: ImageSize | None=None,
        acceleration: Acceleration="local_auto"
    ) -> ImageResponse:
        """
        Create an image given a prompt.

        Parameter:
            prompt (str): Text description of the desired image.
            model (str): Image generation model tag.
            background (str): Set transparency for the background of the generated image(s).
            n (int): Number of images to generate. 
            output_compression (int): The compression level for the generated images in range [0,100].
            output_format (str): The format in which the generated images are returned.
            size (str): The size of the generated images.
            acceleration (Acceleration): Prediction acceleration.

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
            acceleration=acceleration
        )
        # Return
        return result

    def __create_delegate(self, tag: str) -> ImageDelegate:
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        if not predictor:
            raise ValueError(
                f"{tag} cannot be used with OpenAI image API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        # Check that there is only one required input parameter
        signature = predictor.signature
        required_inputs = [param for param in signature.inputs if not param.optional]
        if len(required_inputs) != 1:
            raise ValueError(
                f"{tag} cannot be used with OpenAI image API because "
                "it has more than one required input parameter."
            )
        # Check that the input parameter is `str`
        _, prompt_param = get_parameter(required_inputs, dtype=Dtype.string)
        if prompt_param is None:
            raise ValueError(
                f"{tag} cannot be used with OpenAI image API because "
                "it does not have a valid text prompt input parameter."
            )
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
            prompt: str,
            model: str,
            background: Literal["auto", "transparent", "opaque"] | None,
            n: int | None,
            output_format: Literal["png", "jpeg", "webp", "raw"],
            output_compression: int | None,
            size: ImageSize | None,
            acceleration: Acceleration
        ) -> ImageResponse:
            # Build prediction input map
            requested_width, requested_height = _get_image_size(size)
            input_map = { prompt_param.name: prompt }
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
                acceleration=acceleration
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

def _create_image_data(
    image: Image.Image,
    output_format: Literal["png", "jpeg", "webp", "raw"],
    output_compression: int | None,
) -> ImageData:
    match output_format:
        case "raw":     return ImageData(b64_json=None, image=image)
        case "webp":    raise ValueError(f"webp output format is not yet supported")
        case _:
            mime = f"image/{output_format}"
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