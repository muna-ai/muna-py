#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from muna.beta import Annotations
from PIL import Image
from typing import Annotated

@compile(targets=["x86_64-unknown-linux-gnu"])
def image_generation(
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Description of the image to generate.")
    ],
    *,
    width: Annotated[int, Annotations.ImageWidth(
        description="Image width in pixels.",
        min=32,
        max=1024
    )] = 64,
    height: Annotated[int, Annotations.ImageHeight(
        description="Image height in pixels.",
        min=32,
        max=1024
    )] = 64,
    count: Annotated[int, Annotations.ImageCount(
        description="Number of images to generate.",
        min=1,
        max=4
    )] = 1
) -> list[Image.Image]:
    """
    Signature-only image generation predictor.
    """
    raise NotImplementedError("Signature-only mock predictor.")
