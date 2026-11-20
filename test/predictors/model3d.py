#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Parameter
from muna.beta import Annotations
from PIL import Image
from typing import Annotated

@compile(targets=["x86_64-unknown-linux-gnu"])
def model3d(
    images: Annotated[
        list[Image.Image],
        Annotations.ChatImages(description="Reference images.")
    ],
    *,
    prompt: Annotated[
        str,
        Annotations.ImagePrompt(description="Optional text conditioning.")
    ] = ""
) -> Annotated[
    bytes,
    Parameter.Model3D(description="PLY file containing 3D Gaussian splats.")
]:
    """
    Signature-only 3D generation predictor.
    """
    raise NotImplementedError("Signature-only mock predictor.")
