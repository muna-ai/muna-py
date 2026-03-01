# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from numpy import float32
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

class Audio(BaseModel, **ConfigDict(arbitrary_types_allowed=True, frozen=True)):
    """
    Audio buffer.
    """
    samples: NDArray[float32] = Field(description="Audio samples with shape (F,C).")
    sample_rate: int = Field(description="Audio sample rate (Hz).")
    channel_count: int = Field(description="Audio channel count.")