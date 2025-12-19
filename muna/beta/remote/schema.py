#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import Field
from typing import Literal

from ...types import Prediction, RemoteValue

RemoteAcceleration = Literal[
    "remote_auto",
    "remote_cpu",
    "remote_a10",
    "remote_a100",
    "remote_h200",
    "remote_b200"
]

class RemotePrediction(Prediction):
    """
    Remote prediction.
    """
    results: list[RemoteValue] | None = Field(description="Prediction results.")