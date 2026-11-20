# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from typing import Literal

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device
from ..adapters import (
    ImageInput, ImageResponse, OpenAIDepthAdapter, OpenAIDepthOptions,
    OpenAIImageEditAdapter, OpenAIImageGenerationAdapter, OpenAIImageOptions,
    OpenAIModel3DAdapter, OpenAIModel3DOptions
)

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
        self.__generate_cache: dict[str, OpenAIImageGenerationAdapter] = {}
        self.__edit_cache: dict[
            str, OpenAIImageEditAdapter | OpenAIDepthAdapter | OpenAIModel3DAdapter
        ] = {}

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
        # Ensure adapter
        if model not in self.__generate_cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            adapter = OpenAIImageGenerationAdapter.from_signature(predictor.signature)
            if adapter is None:
                raise ValueError(f"Predictor {model} does not support image generation")
            self.__generate_cache[model] = adapter
        # Predict
        adapter = self.__generate_cache[model]
        width, height = _get_image_size(size)
        inputs = adapter.inputs(
            prompt=prompt,
            n=n,
            width=width,
            height=height
        )
        prediction = self.__predictions.create(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        # Adapt
        options = OpenAIImageOptions(
            format=output_format or "png",
            compression=output_compression or 90
        )
        prediction = self.__predictions.create(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        response = adapter.adapt(prediction, options)
        # Return
        return response

    def edit(
        self,
        *,
        image: ImageInput | list[ImageInput],
        prompt: str | None = None,
        model: str,
        background: Literal["auto", "transparent", "opaque"] | None = None,
        n: int | None = None,
        output_format: Literal["png", "jpeg", "webp", "raw", "npy", "npz", "ply"] | None = None,
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
        # Ensure adapter
        if model not in self.__edit_cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            signature = predictor.signature
            if any(param.denotation == "depth_map" for param in signature.outputs):
                adapter = OpenAIDepthAdapter.from_signature(signature)
            elif any(param.denotation == "3d" for param in signature.outputs):
                adapter = OpenAIModel3DAdapter.from_signature(signature)
            else:
                adapter = OpenAIImageEditAdapter.from_signature(signature)
            if adapter is None:
                raise ValueError(
                    f"Predictor {model} does not support image editing, depth estimation, or 3D generation"
                )
            self.__edit_cache[model] = adapter
        # Predict
        adapter = self.__edit_cache[model]
        width, height = _get_image_size(size)
        inputs = adapter.inputs(
            images=image if isinstance(image, list) else [image],
            prompt=prompt,
            n=n,
            width=width,
            height=height
        )
        # Select and validate the encoding before executing the predictor.
        match adapter.kind:
            case "openai.images.edit.model3d":
                if background is not None or output_compression is not None:
                    raise ValueError("3D output does not accept background or output_compression")
                options = OpenAIModel3DOptions(format=output_format or "ply")
            case "openai.images.edit.depth":
                if background is not None or output_compression is not None:
                    raise ValueError("Depth output does not accept background or output_compression")
                options = OpenAIDepthOptions(format=output_format or "npz")
            case "openai.images.edit":
                options = OpenAIImageOptions(
                    format=output_format or "png",
                    compression=output_compression or 90
                )
        prediction = self.__predictions.create(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        response = adapter.adapt(prediction, options)
        # Return
        return response

def _get_image_size(size: ImageSize | None) -> tuple[int | None, int | None]:
    try:
        w, h = size.split("x", 1)
        return int(w), int(h)
    except (ValueError, AttributeError):
        return None, None
