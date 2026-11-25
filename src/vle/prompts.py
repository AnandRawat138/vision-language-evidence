"""Loads version-controlled prompts from configs/prompts.yaml."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_FILE = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"


@lru_cache(maxsize=1)
def _all() -> dict:
    return yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))


def get_prompt(prompt_id: str) -> dict:
    p = dict(_all()[prompt_id])
    p["id"] = prompt_id
    p["sha1"] = hashlib.sha1((str(p.get("system")) + p["text"]).encode()).hexdigest()[:10]
    return p


def render(prompt_id: str, **kw: object) -> str:
    p = get_prompt(prompt_id)
    text = p["text"]
    for k, v in kw.items():
        text = text.replace("{" + k + "}", str(v))
    return text
