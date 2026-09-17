# 
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

from __future__ import annotations

def prediction_error_to_exception(error: str) -> Exception:
    """
    Classify a predictor's `prediction.error` string.

    Predictors raise Python exceptions; the runtime formats them as
    `Type: message`, optionally preceded by a traceback and a chain of causes
    (innermost first). `ValueError` and `TypeError` are re-raised as
    themselves carrying only the message: the traceback belongs in server
    logs, not in a caller-facing error. Anything else is an opaque
    `RuntimeError` with the full text. Mirrors `MunaError::from_prediction_error`
    in muna-rs, which maps the same two classes to `InvalidInput` (a 400).
    """
    # The outermost exception is the last one in the chain; only its first
    # non-traceback line carries the `Type: message` header.
    outermost = error.rsplit(_CAUSE_SEPARATOR, 1)[-1]
    header = next((
        line
        for line in outermost.splitlines()
        if line and not line.startswith(" ") and not line.startswith("Traceback")
    ), "")
    kind, sep, message = header.partition(": ")
    if sep and kind in _CALLER_FAULTS:
        return _CALLER_FAULTS[kind](message.strip())
    return RuntimeError(error)

_CAUSE_SEPARATOR = "\n\nThe above exception was the direct cause of the following exception:\n\n"
# Python's caller-fault exceptions. A predictor raising one of these before
# its first output is reporting bad input, not a broken model.
_CALLER_FAULTS: dict[str, type[Exception]] = {
    "ValueError": ValueError,
    "TypeError": TypeError
}