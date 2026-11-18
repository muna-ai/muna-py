#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Parameter
from muna.beta import Annotations
from numpy import ndarray
from PIL import Image
from typing import Annotated

@compile(targets=["x86_64-unknown-linux-gnu"])
def depth(
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Instructions for depth estimation.")
    ],
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Images whose depth maps should be estimated.")
    ]
) -> Annotated[
    list[ndarray],
    Parameter.DepthMap(description="One depth map per input image.")
]:
    """
    Signature-only depth estimation predictor.
    """
    raise NotImplementedError("Signature-only mock predictor.")
