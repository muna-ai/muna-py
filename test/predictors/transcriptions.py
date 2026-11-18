#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Parameter
from muna.beta import Annotations
from numpy import ndarray
from typing import Annotated

@compile(targets=["x86_64-unknown-linux-gnu"])
def transcription(
    audio: Annotated[ndarray, Parameter.Audio(
        description="Mono audio samples to transcribe.",
        sample_rate=16000
    )],
    *,
    language: Annotated[
        str,
        Annotations.TranscriptionLanguage(description="Spoken language as an ISO 639-1 code.")
    ] = "en"
) -> str:
    """
    Signature-only speech transcription predictor.
    """
    raise NotImplementedError("Signature-only mock predictor.")
