# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from functools import cached_property
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Any, Literal

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
