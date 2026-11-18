#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import compile, Parameter
from muna.beta import Annotations
from numpy import ndarray
from typing import Annotated

@compile(targets=["x86_64-unknown-linux-gnu"])
def embeddings(
    texts: list[str],
    *,
    dimensions: Annotated[int, Annotations.EmbeddingDims(
        description="Number of embedding dimensions.",
        min=128,
        max=1024
    )] = 3
) -> Annotated[
    ndarray,
    Parameter.Embedding(description="Embedding matrix with one row per input text.")
]:
    """
    Signature-only text embedding predictor.
    """
    raise NotImplementedError("Signature-only mock predictor.")
