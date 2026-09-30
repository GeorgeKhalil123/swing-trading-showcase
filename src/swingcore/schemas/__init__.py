"""Responsibility: load the JSON Schemas shipped next to this file and validate agent output.

Every agent reply is validated before it is used. `validation_error` returns the message the runtime
feeds back on the single allowed retry; `validate` is the fail-loud form used once an object is
meant to be known-good (e.g. a fixture or a stored run manifest).

Shipped here: a trimmed `run_manifest` and two toy agent schemas (`toy_trend`, `toy_tone`) written
for the demo. The private system's research, synthesis and decision schemas are not included.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_DIR = Path(__file__).resolve().parent
SCHEMA_NAMES = ("run_manifest", "toy_trend", "toy_tone")


class SchemaValidationError(ValueError):
    """Raised when an object does not match its declared schema."""


@cache
def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMA_DIR / f"{name}.json"
    if not path.exists():
        raise KeyError(f"unknown schema '{name}'; known: {', '.join(SCHEMA_NAMES)}")
    data: dict[str, Any] = json.loads(path.read_text())
    return data


@cache
def _validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validation_error(name: str, obj: Any) -> str | None:
    """Return a single human-readable error string, or None when `obj` is valid."""
    errors = sorted(_validator(name).iter_errors(obj), key=lambda e: list(e.absolute_path))
    if not errors:
        return None
    parts = [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors[:5]]
    return f"schema '{name}' rejected the output: " + "; ".join(parts)


def validate(name: str, obj: Any, context: str = "") -> None:
    err = validation_error(name, obj)
    if err is not None:
        raise SchemaValidationError(f"{context + ': ' if context else ''}{err}")
