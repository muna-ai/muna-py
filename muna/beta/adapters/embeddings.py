#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from base64 import b64encode
from numpy import ndarray
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from ...types import Dtype, Parameter, Prediction, Signature, Value
from ._adapter import Adapter
from ._utils import decode, get_parameter, result, validate_inputs, INTEGERS

class Embedding(BaseModel):
    object: Literal["embedding"]
    embedding: list[float] | str
    index: int

class EmbeddingCreateResponse(BaseModel):
    class Usage(BaseModel):
        prompt_tokens: int
        total_tokens: int
    object: Literal["list"]
    model: str
    data: list[Embedding]
    usage: Usage

class OpenAIEmbeddingsOptions(BaseModel, **ConfigDict(extra="forbid", frozen=True, strict=True)):
    """
    Encoding for each vector in the embeddings response.
    """
    encoding_format: Literal["float", "base64"] = "float"

class OpenAIEmbeddingsAdapter(Adapter[Prediction, OpenAIEmbeddingsOptions, EmbeddingCreateResponse]):
    """
    Prepare text inputs and adapt an embedding matrix with optional token usage.
    """
    kind: Literal["openai.embeddings.create"] = Field("openai.embeddings.create", init=False)
    input: Parameter
    output: int
    dimensions: Parameter | None = None
    usage: int | None = None

    @classmethod
    def from_signature(cls, signature: Signature) -> OpenAIEmbeddingsAdapter | None:
        """
        Create an OpenAI embeddings adapter from a predictor signature.
        """
        signature = signature.model_copy(deep=True)
        # Text input
        _, input = get_parameter(
            [param for param in signature.inputs if not param.optional],
            dtype=Dtype.list
        )
        if input is None:
            return None
        # Matryoshka dimensions
        _, dimensions = get_parameter(
            signature.inputs,
            dtype=INTEGERS,
            denotation="openai.embeddings.dims"
        )
        # Validate inputs
        if not validate_inputs(signature, [input, dimensions]):
            return None
        # Embedding matrix
        output, _ = get_parameter(
            signature.outputs,
            dtype=Dtype.float32,
            denotation="embedding"
        )
        if output is None:
            return None
        # Token usage
        _, usage = get_parameter(
            [param for param in signature.outputs if (param.value_schema or {}).get("title") == "Usage"],
            dtype=Dtype.dict
        )
        return cls(
            input=input,
            output=output,
            dimensions=dimensions,
            usage=signature.outputs.index(usage) if usage is not None else None
        )

    def inputs(
        self,
        *,
        input: str | list[str],
        dimensions: int | None = None
    ) -> dict[str, Value]:
        """
        Create prediction inputs from an embeddings request.
        """
        inputs = { self.input.name: [input] if isinstance(input, str) else input }
        if self.dimensions is not None and dimensions is not None:
            inputs[self.dimensions.name] = dimensions
        return inputs

    def adapt(
        self,
        prediction: Prediction,
        options: OpenAIEmbeddingsOptions | None = None
    ) -> EmbeddingCreateResponse:
        options = options or OpenAIEmbeddingsOptions()
        matrix = decode(result(prediction, self.output), Dtype.float32)
        if not isinstance(matrix, ndarray) or matrix.dtype != "float32" or matrix.ndim != 2:
            raise ValueError("Embeddings output must be a two-dimensional float32 matrix")
        usage = (
            EmbeddingCreateResponse.Usage.model_validate(decode(result(prediction, self.usage), Dtype.dict))
            if self.usage is not None
            else EmbeddingCreateResponse.Usage(prompt_tokens=0, total_tokens=0)
        )
        return EmbeddingCreateResponse(
            object="list",
            model=prediction.tag,
            data=[Embedding(
                object="embedding",
                index=index,
                # OpenAI base64 vectors contain raw little-endian float32 bytes,
                # rather than the NPZ representation used on the Muna wire.
                embedding=(
                    b64encode(vector.astype("<f4", copy=False).tobytes()).decode("ascii")
                    if options.encoding_format == "base64"
                    else vector.tolist()
                )
            ) for index, vector in enumerate(matrix)],
            usage=usage
        )
