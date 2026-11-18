# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Mapping

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Device
from ..adapters.system_one import (
    Content, Question, SystemOneResponse, TypeSafeSystemOneAdapter,
    TypeSafeSystemOneOptions, _QUESTION
)

class TypeSafeClient:
    """
    Experimental TypeSafe-compatible System One client.
    """

    def __init__(
        self,
        predictors: PredictorService,
        predictions: PredictionService
    ):
        self.__predictors = predictors
        self.__predictions = predictions
        self.__cache: dict[str, TypeSafeSystemOneAdapter] = {}

    def system_one(
        self,
        state: Content,
        questions: Mapping[str, Question | dict],
        *,
        model: str,
        acceleration: Acceleration="local_auto",
        devices: list[Device | str] | None=None
    ) -> SystemOneResponse:
        """
        Evaluate a state against a map of typed questions.

        Parameters:
            state (str | dict | list): Content that all questions refer to.
            questions (dict): Typed questions keyed by caller-chosen ids. Ids are never sent to the model.
            model (str): Decision model tag.
            acceleration (Acceleration): Prediction acceleration.
            devices (list): Local compute devices.

        Returns:
            SystemOneResponse: System one response.
        """
        # Ensure adapter
        if model not in self.__cache:
            predictor = self.__predictors.retrieve(model)
            if predictor is None:
                raise ValueError(f"Predictor {model} could not be found")
            adapter = TypeSafeSystemOneAdapter.from_signature(predictor.signature)
            if adapter is None:
                raise ValueError(f"Predictor {model} does not support TypeSafe System One")
            self.__cache[model] = adapter
        # Predict
        adapter = self.__cache[model]
        inputs = adapter.inputs(
            state=state,
            questions=questions
        )
        prediction = self.__predictions.create(
            tag=model,
            inputs=inputs,
            acceleration=acceleration,
            devices=devices,
            passthrough=True
        )
        # Adapt
        options = TypeSafeSystemOneOptions()
        response = adapter.adapt(prediction, options)
        # Return
        return response
