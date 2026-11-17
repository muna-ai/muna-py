# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from pydantic import BaseModel, Field
from typing import Literal

from ...services import PredictorService
from ...types import Signature
from ..utils import rfc3339_to_unix, split_tag

class Model(BaseModel):
    """
    OpenAI model object: exactly the four fields OpenAI documents.
    """
    id: str = Field(description="Model identifier (the predictor tag).")
    object: Literal["model"] = Field("model", description="Object type, always `model`.")
    created: int = Field(description="Creation time in Unix seconds.")
    owned_by: str = Field(description="Model owner, the tag's `@owner`.")

class ModelService:
    """
    Describe models.
    """

    def __init__(self, predictors: PredictorService):
        self.__predictors = predictors

    def retrieve(self, model: str) -> Model:
        """
        Retrieve a model.

        Parameters:
            model (str): Predictor tag.
        """
        predictor = self.__predictors.retrieve(model)
        if not predictor:
            raise ValueError(
                f"{model} cannot be retrieved with the OpenAI models API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        return _create_model(
            predictor.tag,
            predictor.signature,
            created=rfc3339_to_unix(predictor.created)
        )

def _create_model(
    tag: str,
    signature: Signature,
    *,
    created: int | None
) -> Model:
    """
    Build the model object for a predictor.

    The signature is taken for parity with the Anthropic projection (which
    reads limits from it); OpenAI's object carries none, so it is unused
    today. `created` is the Unix time the model became available.
    """
    owner, _ = split_tag(tag)
    return Model(
        id=tag,
        created=created or 0,
        owned_by=owner
    )
