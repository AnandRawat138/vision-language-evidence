"""Structured-output validation (schema enforcement and conservative repair).

``parse_structured`` is the framework path: extract the JSON object, apply
minimal syntactic repairs, then validate/coerce with Pydantic.
``parse_naive`` is the ablation path ("without structured validation"):
plain ``json.loads`` of the raw text, no repair, no schema coercion.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from vle.schemas import VLMStructuredOutput

ParseStatus = Literal["valid", "repaired", "failed"]
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_object(text: str) -> str | None:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # truncated object


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)              # trailing commas
    s = re.sub(r"(?<=[{,\s])'([^'\"]+)'\s*:", r'"\1":', s)  # single-quoted keys
    s = s.replace("None", "null").replace("True", "true").replace("False", "false")
    # close truncated structures
    stack = []
    in_str, esc = False, False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append(c)
        elif c in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    s = re.sub(r",\s*$", "", s.rstrip())
    s = re.sub(r':\s*$', ": null", s)
    for c in reversed(stack):
        s += "}" if c == "{" else "]"
    return s


def parse_structured(text: str, model: type[BaseModel] = VLMStructuredOutput) -> tuple[BaseModel | None, ParseStatus, str]:
    obj = _extract_object(text)
    if obj is None:
        return None, "failed", "no JSON object"
    for attempt, cand in enumerate((obj, _repair(obj))):
        try:
            data = json.loads(cand)
        except json.JSONDecodeError as e:
            err = f"json: {e}"
            continue
        try:
            return model.model_validate(data), ("valid" if attempt == 0 else "repaired"), ""
        except ValidationError as e:
            err = f"schema: {e.errors()[0]['msg']}"
            continue
    return None, "failed", err


def parse_naive(text: str) -> tuple[dict | None, ParseStatus]:
    """Minimal parser: strip a Markdown code fence, then json.loads. No repair, no schema coercion."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    try:
        d = json.loads(text.strip())
        return (d, "valid") if isinstance(d, dict) else (None, "failed")
    except json.JSONDecodeError:
        return None, "failed"


def find_value_span(raw: str, value: str, search_from: int = 0) -> tuple[int, int] | None:
    """Character span of a quoted string value in the raw model output (for token confidence)."""
    if not value:
        return None
    i = raw.lower().find(f'"{value.lower()}"', search_from)
    if i < 0:
        i = raw.lower().find(value.lower(), search_from)
        if i < 0:
            return None
        return i, i + len(value)
    return i + 1, i + 1 + len(value)
