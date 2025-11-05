#
#   Muna
#   Copyright © 2025 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, Field
from typing import Literal

from ...types import Dtype, Prediction

RemoteAcceleration = Literal["remote_auto", "remote_cpu", "remote_a40", "remote_a100"]

class RemoteValue(BaseModel):
    """
    Remote value.
    """
    data: str | None = Field(description="Value URL. This is a remote or data URL.")
    type: Dtype = Field(description="Value type.")
    shape: list[int] | None = Field(default=None, description="Value shape. This is `None` if shape information is not available or applicable.")

class RemotePrediction(Prediction):
    """
    Remote prediction.
    """
    results: list[RemoteValue] | None = Field(description="Prediction results.")