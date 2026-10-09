# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from collections.abc import Callable, Mapping
from pydantic import TypeAdapter

from ...services import PredictorService, PredictionService
from ...types import Acceleration, Dtype
from ..annotations import get_parameter
from ..errors import prediction_error_to_exception
from .schema import Content, Question, SystemOneResponse

SystemOneDelegate = Callable[..., SystemOneResponse]

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
        self.__cache = dict[str, SystemOneDelegate]()

    def system_one(
        self,
        state: Content,
        questions: Mapping[str, Question | dict],
        *,
        model: str,
        acceleration: Acceleration="local_auto"
    ) -> SystemOneResponse:
        """
        Evaluate a state against a map of typed questions.

        Parameters:
            state (str | dict | list): Content that all questions refer to.
            questions (dict): Typed questions keyed by caller-chosen ids. Ids are never sent to the model.
            model (str): Decision model tag.
            acceleration (Acceleration): Prediction acceleration.
        """
        # Validate request
        if not isinstance(state, (str, dict, list)):
            raise ValueError("`state` must be a string, object, or array.")
        if not questions:
            raise ValueError("`questions` must contain at least one question.")
        questions = { id: _QUESTION.validate_python(q) for id, q in questions.items() }
        # Ensure we have a delegate
        if model not in self.__cache:
            self.__cache[model] = self.__create_delegate(model)
        # Make prediction
        delegate = self.__cache[model]
        result = delegate(
            state=state,
            questions=questions,
            model=model,
            acceleration=acceleration
        )
        # Return
        return result

    def __create_delegate(self, tag: str) -> SystemOneDelegate:
        # Retrieve predictor
        predictor = self.__predictors.retrieve(tag)
        if not predictor:
            raise ValueError(
                f"{tag} cannot be used with TypeSafe System One API because "
                "the predictor could not be found. Check that your access key "
                "is valid and that you have access to the predictor."
            )
        # Check that there are exactly two required input parameters
        required_inputs = [
            param
            for param in predictor.signature.inputs
            if not param.optional
        ]
        if len(required_inputs) != 2:
            raise ValueError(
                f"{tag} cannot be used with TypeSafe System One API because "
                "it does not have exactly two required input parameters."
            )
        # Get the state and questions input parameters
        _, state_param = get_parameter(
            required_inputs,
            dtype=Dtype.list,
            denotation="typesafe.systemone.state"
        )
        _, questions_param = get_parameter(
            required_inputs,
            dtype=Dtype.dict,
            denotation="typesafe.systemone.questions"
        )
        if state_param is None or questions_param is None:
            raise ValueError(
                f"{tag} cannot be used with TypeSafe System One API because "
                "it does not have a `list` input with a `typesafe.systemone.state` "
                "denotation and a `dict` input with a `typesafe.systemone.questions` denotation."
            )
        # Get the response output parameter index
        response_param_idx = next((
            idx
            for idx, param in enumerate(predictor.signature.outputs)
            if (
                param.dtype == Dtype.dict and
                param.value_schema and
                param.value_schema.get("title") == "SystemOneResponse"
            )
        ), None)
        if response_param_idx is None:
            raise ValueError(
                f"{tag} cannot be used with TypeSafe System One API because "
                "it has no `SystemOneResponse` output."
            )
        # Define delegate
        def delegate(
            *,
            state: Content,
            questions: dict[str, Question],
            model: str,
            acceleration: Acceleration
        ) -> SystemOneResponse:
            # Build prediction input map
            input_map = {
                state_param.name: state if isinstance(state, list) else [state],
                questions_param.name: {
                    id: question.model_dump(exclude_none=True)
                    for id, question in questions.items()
                }
            }
            # Create prediction
            prediction = self.__predictions.create(
                tag=model,
                inputs=input_map,
                acceleration=acceleration
            )
            # Check for error
            if prediction.error:
                raise prediction_error_to_exception(prediction.error)
            # Return
            output = prediction.results[response_param_idx]
            if not isinstance(output, dict):
                raise RuntimeError(f"{tag} returned object of type {type(output)} instead of a System One response")
            return SystemOneResponse.model_validate({ **output, "model": model })
        # Return
        return delegate

_QUESTION = TypeAdapter(Question)
