#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations
from collections.abc import Mapping
from functools import cached_property
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing import Annotated, Any, Literal

from ...types import Dtype, Parameter, Prediction, Signature, Value
from ._adapter import Adapter
from ._utils import decode, get_parameter, result, validate_inputs

Content = str | dict[str, Any] | list[Any]

class Noul(BaseModel):
    """
    Yes/no question, answered with the probability of yes.
    """
    type: Literal["noul"] = "noul"
    instructions: Content
    criteria: dict[Literal["true", "false"], Content | None] | None = None

class Choice(BaseModel):
    """
    Question that picks one of the named options in `criteria`.
    """
    type: Literal["choice"] = "choice"
    instructions: Content
    criteria: dict[str, Content | None] = Field(min_length=1, max_length=255)

class Score(BaseModel):
    """
    Question that rates on the ordered levels in `criteria`, low to high.
    """
    type: Literal["score"] = "score"
    instructions: Content
    criteria: list[Content] = Field(min_length=2, max_length=10)

Question = Annotated[
    Noul    |
    Choice  |
    Score,
    Field(discriminator="type")
]

class NoulAnswer(BaseModel, **ConfigDict(frozen=True)):
    """
    Answer to a noul question.
    """
    type: Literal["noul"] = "noul"
    noul: float

class ChoiceAnswer(BaseModel, **ConfigDict(frozen=True)):
    """
    Answer to a choice question.
    """
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]

class ScoreAnswer(BaseModel, **ConfigDict(frozen=True)):
    """
    Answer to a score question. `legend` and `probabilities` are keyed by level.
    """
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[int, Content]
    probabilities: dict[int, float]

Answer = Annotated[
    NoulAnswer      |
    ChoiceAnswer    |
    ScoreAnswer,
    Field(discriminator="type")
]

class SystemOneResponse(BaseModel, **ConfigDict(frozen=True, title="SystemOneResponse")):
    """
    One answer per question, keyed by the question ids.
    """
    class Usage(BaseModel, **ConfigDict(frozen=True)):
        """
        Token usage for a System One request.
        """
        input_tokens: int | None = None
        output_tokens: int | None = None
    model: str
    answers: dict[str, Answer]
    usage: Usage

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {
            k: v
            for k, v in self.answers.items()
            if v.type == "noul"
        }

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {
            k: v
            for k, v in self.answers.items()
            if v.type == "choice"
        }

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {
            k: v
            for k, v in self.answers.items()
            if v.type == "score"
        }

class TypeSafeSystemOneOptions(BaseModel, **ConfigDict(extra="forbid", frozen=True)):
    """
    System One response options.
    """

class TypeSafeSystemOneAdapter(Adapter[Prediction, TypeSafeSystemOneOptions, SystemOneResponse]):
    """
    Prepare typed decision questions and adapt their answers and token usage.
    """
    kind: Literal["typesafe.system_one"] = Field("typesafe.system_one", init=False)
    state: Parameter
    questions: Parameter
    output: int

    @classmethod
    def from_signature(cls, signature: Signature) -> TypeSafeSystemOneAdapter | None:
        """
        Create a TypeSafe system one adapter from a predictor signature.
        """
        signature = signature.model_copy(deep=True)
        required = [param for param in signature.inputs if not param.optional]
        # State input
        _, state = get_parameter(
            required,
            dtype=Dtype.list,
            denotation="typesafe.systemone.state"
        )
        if state is None:
            return None
        # Question map
        _, questions = get_parameter(
            required,
            dtype=Dtype.dict,
            denotation="typesafe.systemone.questions"
        )
        if questions is None:
            return None
        if not validate_inputs(signature, [state, questions]):
            return None
        # Typed response
        _, response = get_parameter(
            [param for param in signature.outputs if (param.value_schema or {}).get("title") == "SystemOneResponse"],
            dtype=Dtype.dict
        )
        if response is None:
            return None
        # Return
        return cls(
            state=state,
            questions=questions,
            output=signature.outputs.index(response)
        )

    def inputs(
        self,
        *,
        state: Content,
        questions: Mapping[str, Question | dict]
    ) -> dict[str, Value]:
        if not isinstance(state, (str, dict, list)):
            raise ValueError("`state` must be a string, object, or array.")
        if not questions:
            raise ValueError("`questions` must contain at least one question.")
        return {
            self.state.name: state if isinstance(state, list) else [state],
            self.questions.name: {
                id: _QUESTION.validate_python(question).model_dump(exclude_none=True)
                for id, question in questions.items()
            }
        }

    def adapt(
        self,
        prediction: Prediction,
        options: TypeSafeSystemOneOptions | None = None
    ) -> SystemOneResponse:
        output = decode(result(prediction, self.output), Dtype.dict)
        if not isinstance(output, dict):
            raise ValueError("System One output must be a dictionary")
        return SystemOneResponse.model_validate({**output, "model": prediction.tag})

_QUESTION = TypeAdapter(Question)
