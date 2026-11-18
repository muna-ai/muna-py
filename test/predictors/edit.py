#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile
from muna.beta import Annotations
from PIL import Image
from typing import Annotated

@compile(targets=["x86_64-unknown-linux-gnu"])
def image_edit(
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Description of the desired image edit.")
    ],
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Input images to edit.")
    ]
) -> list[Image.Image]:
    """
    Signature-only image editing predictor.
    """
    raise NotImplementedError("Signature-only mock predictor.")
