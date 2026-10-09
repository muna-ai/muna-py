# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from muna import Muna
from muna.beta.typesafe import Choice, Noul, Score, SystemOneResponse
from muna.beta.typesafe.typesafe import _QUESTION
from muna.types import Signature
from pydantic import ValidationError
from pytest import mark, raises

def _signature(inputs: list[dict], outputs: list[dict]) -> Signature:
    return Signature.model_validate({ "inputs": inputs, "outputs": outputs })

_DECISION_SIGNATURE = _signature(
    [
        { "name": "context", "dtype": "list", "denotation": "typesafe.systemone.state" },
        { "name": "schema", "dtype": "dict", "denotation": "typesafe.systemone.questions" }
    ],
    [{ "name": "result", "dtype": "dict", "schema": { "title": "SystemOneResponse" } }]
)

def test_score_requires_two_levels():
    with raises(ValidationError):
        Score(instructions="Rate.", criteria=["only"])

def test_choice_requires_an_option():
    with raises(ValidationError):
        Choice(instructions="Pick.", criteria={})

def test_question_dicts_validate_by_type():
    question = _QUESTION.validate_python({
        "type": "noul",
        "instructions": "Escalate?",
        "criteria": { "true": "Needs a human", "false": "Bot can handle it" }
    })
    assert isinstance(question, Noul)
    with raises(ValidationError):
        _QUESTION.validate_python({ "type": "noul", "instructions": "Yes?", "criteria": { "maybe": "unsure" } })

def test_parse_response_views():
    response = SystemOneResponse.model_validate({
        "model": "@bespokelabs/nimble-9b",
        "answers": {
            "route": {
                "type": "choice", "choice": "billing", "confidence": 0.99,
                "probabilities": { "billing": 0.99, "bug": 0.01 }
            },
            "escalate": { "type": "noul", "noul": 0.94 },
            "urgency": {
                "type": "score", "score": 1.6, "confidence": 0.6,
                "legend": { "0": "routine", "1": "today", "2": "urgent" },
                "probabilities": { "0": 0.1, "1": 0.2, "2": 0.7 }
            }
        },
        "usage": { "input_tokens": 120 }
    })
    assert list(response.answers) == ["route", "escalate", "urgency"]
    assert response.choices["route"].choice == "billing"
    assert response.nouls["escalate"].noul == 0.94
    assert response.scores["urgency"].probabilities[2] == 0.7
    assert response.scores["urgency"].legend[0] == "routine"
    assert response.usage.output_tokens is None

@mark.skip(reason="requires a published System One decision predictor")
def test_system_one():
    typesafe = Muna().beta.typesafe
    response = typesafe.system_one(
        "Customer: I was charged twice and nobody has replied for 3 days.",
        {
            "route": Choice(
                instructions="Where should this go?",
                criteria={ "billing": "money", "bug": "broken", "account": "login" }
            ),
            "urgency": Score(
                instructions="How urgent is this?",
                criteria=["routine", "today", "urgent", "critical"]
            ),
            "escalate": Noul(instructions="Escalate to a human now?")
        },
        model="@bespokelabs/nimble-9b",
        acceleration="local_auto"
    )
    assert set(response.answers) == { "route", "urgency", "escalate" }
    print(response.model_dump_json(indent=2))
