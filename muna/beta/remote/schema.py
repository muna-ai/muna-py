#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import AliasChoices, BaseModel, Field
from typing import Literal

from ...types import Dtype, Prediction

RemoteAcceleration = Literal[
    "remote_auto",
    "remote_cpu",
    "remote_a10",
    "remote_a100",
    "remote_h200",
    "remote_b200"
]

class RemoteValue(BaseModel):
    """
    Remote value.
    """
    data: str | None = Field(description="Value URL. This is a remote or data URL.")
    dtype: Dtype = Field(description="Value type.", validation_alias=AliasChoices("dtype", "type"))

class RemotePrediction(Prediction):
    """
    Remote prediction.
    """
    results: list[RemoteValue] | None = Field(default=None, description="Prediction results.")