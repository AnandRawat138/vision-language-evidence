"""Module B: natural-language observation structuring.

Two interchangeable structurers produce the same Observation records:

* ``RuleStructurer``  — deterministic lexicon + clause-scoped cue detection
  (hedging, negation), colour-adjective attachment, exact character spans.
* ``LLMStructurer``   — the VLM used text-only with prompt ``narrative-v1``;
  its output is validated, and every item must be supported by a quote that
  occurs in the original text (unsupported items are dropped and counted).

Invariants (tested): the original text is stored unchanged; a hedged or
negated statement is never emitted as ``definite``.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, field_validator

from vle.lexicon import (ACCESSORIES, CARRIED, CLOTHING, COLORS, HEDGE_CUES, NEGATION_CUES,
                         SCENE_PLACES, canon_color, normalize_phrase)
from vle.parsing import parse_structured
from vle.schemas import Category, Certainty, Observation, Provenance, UncertaintyInfo

_CLAUSE_SPLIT = re.compile(r"[.;!?]|,\s*(?:but|although|though)\b|\bbut\b")
_TABLES = [(Category.CLOTHING, CLOTHING), (Category.ACCESSORY, ACCESSORIES), (Category.CARRIED, CARRIED),
           (Category.SCENE, SCENE_PLACES)]
LIGHTING_CUES = ["dark", "night", "dim", "bright", "sunny", "daylight", "evening", "shadow", "lit", "lamp", "streetlight"]


@dataclass
class NarrativeResult:
    narrative_id: str
    original_text: str
    observations: list[Observation]
    missing: list[str]
    ambiguous_spans: list[tuple[int, int, str]] = field(default_factory=list)
    unsupported_dropped: int = 0
    parse_status: str = "n/a"


def _clauses(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _CLAUSE_SPLIT.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [s for s in spans if s[1] > s[0]]


def _has_cue(clause: str, cues: list[str]) -> bool:
    c = " " + clause.lower() + " "
    return any((cue if cue.startswith("-") else " " + cue) in c for cue in cues)


def _form_regex(table: dict[str, list[str]]) -> list[tuple[re.Pattern, str]]:
    pats = []
    for canon, forms in table.items():
        for f in sorted(set(forms + [canon]), key=len, reverse=True):
            pats.append((re.compile(r"\b" + re.escape(f) + r"(?:s|es)?\b", re.I), canon))
    pats.sort(key=lambda p: -len(p[0].pattern))
    return pats


_PATTERNS = {cat: _form_regex(tab) for cat, tab in _TABLES}
_COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")(?:[- ]ish)?\b", re.I)


class RuleStructurer:
    name = "rule-v1"

    def structure(self, text: str, narrative_id: str | None = None, source_id: str = "narrative") -> NarrativeResult:
        nid = narrative_id or uuid.uuid4().hex[:12]
        obs: list[Observation] = []
        taken: list[tuple[int, int]] = []
        ambiguous: list[tuple[int, int, str]] = []
        clauses = _clauses(text)
        for cat, pats in _PATTERNS.items():
            for pat, canon in pats:
                for m in pat.finditer(text):
                    s, e = m.span()
                    if any(not (e <= a or s >= b) for a, b in taken):
                        continue
                    taken.append((s, e))
                    cs, ce = next(((a, b) for a, b in clauses if a <= s < b), (0, len(text)))
                    clause = text[cs:ce]
                    pre = text[cs:s]
                    if _has_cue(pre[-40:], NEGATION_CUES):
                        cert = Certainty.NEGATED
                    elif _has_cue(clause, HEDGE_CUES) or re.search(r"\bor\b", clause, re.I):
                        cert = Certainty.HEDGED
                        ambiguous.append((cs, ce, clause.strip()))
                    else:
                        cert = Certainty.DEFINITE
                    color = None
                    if cat == Category.CLOTHING:
                        window = text[max(cs, s - 30):s]
                        cols = _COLOR_RE.findall(window)
                        if cols:
                            color = canon_color(cols[-1])
                            if re.search(r"\bor\b", window, re.I) and len(cols) > 1:
                                cert = Certainty.HEDGED if cert == Certainty.DEFINITE else cert
                    obs.append(Observation(
                        category=cat, label=canon, canonical=canon, raw_text=text[s:e], attribute=color,
                        provenance=Provenance(source_type="narrative", source_id=source_id,
                                              prompt_version=self.name, evidence_ref=f"{nid}:{s}-{e}"),
                        uncertainty=UncertaintyInfo(linguistic_certainty=cert,
                                                    needs_review=cert != Certainty.DEFINITE,
                                                    review_reasons=[f"linguistic:{cert.value}"] if cert != Certainty.DEFINITE else [])))
        return NarrativeResult(nid, text, obs, _missing(text, obs), ambiguous)


def _missing(text: str, obs: list[Observation]) -> list[str]:
    cats = {o.category for o in obs}
    miss = []
    if Category.CLOTHING not in cats:
        miss.append("clothing")
    if Category.ACCESSORY not in cats:
        miss.append("accessories")
    if Category.CARRIED not in cats:
        miss.append("carried_objects")
    if Category.SCENE not in cats:
        miss.append("scene")
    if not any(re.search(r"\b" + c + r"\b", text, re.I) for c in LIGHTING_CUES):
        miss.append("lighting")
    return miss


# ------------------------------------------------------------------ LLM ----
class _NItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    color: str | None = None
    certainty: str = "definite"
    quote: str = ""

    @field_validator("certainty", mode="before")
    @classmethod
    def _c(cls, v: object) -> str:
        s = str(v or "definite").lower()
        return s if s in {"definite", "hedged", "negated"} else "hedged"


class _NScene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    place: str | None = None
    setting: str | None = None
    lighting: str | None = None


class NarrativeLLMOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    clothing: list[_NItem] = []
    accessories: list[_NItem] = []
    carried_objects: list[_NItem] = []
    objects: list[_NItem] = []
    scene: _NScene = _NScene()
    missing: list[str] = []

    @field_validator("clothing", "accessories", "carried_objects", "objects", mode="before")
    @classmethod
    def _items(cls, v: object) -> list:
        if not v:
            return []
        return [({"name": x} if isinstance(x, str) else x) for x in v if x]  # type: ignore[union-attr]


class LLMStructurer:
    """Uses an InferenceBackend text-only. Grounding check: each item's quote (or name) must occur in the text."""
    name = "llm-narrative-v1"

    def __init__(self, backend, prompt_id: str = "narrative-v1"):  # type: ignore[no-untyped-def]
        self.backend, self.prompt_id = backend, prompt_id

    def build_requests(self, items: list[tuple[str, str]]):  # type: ignore[no-untyped-def]
        from vle.inference.base import InferenceRequest
        from vle.prompts import get_prompt, render
        p = get_prompt(self.prompt_id)
        return [InferenceRequest(prompt_id=self.prompt_id, text=render(self.prompt_id, narrative=t.replace('"', "'")),
                                 system=p.get("system"), max_new_tokens=p["max_new_tokens"], image_ids=[nid],
                                 meta={"narrative_id": nid}) for nid, t in items]

    def from_output(self, text: str, raw_output: str, narrative_id: str, source_id: str = "narrative") -> NarrativeResult:
        from vle.lexicon import map_term
        parsed, status, _ = parse_structured(raw_output, NarrativeLLMOutput)
        obs: list[Observation] = []
        dropped = 0
        low = text.lower()
        if parsed is None:
            return NarrativeResult(narrative_id, text, [], [], parse_status=status)
        groups = [(Category.CLOTHING, parsed.clothing, "clothing"), (Category.ACCESSORY, parsed.accessories, "accessories"),
                  (Category.CARRIED, parsed.carried_objects, "carried"), (Category.OBJECT, parsed.objects, "coco")]
        for cat, items, table in groups:
            for it in items:
                anchor = (it.quote or it.name).lower().strip()
                pos = low.find(anchor) if anchor else -1
                if pos < 0:
                    pos = low.find(it.name.lower().strip())
                    anchor = it.name.lower().strip()
                if pos < 0:
                    dropped += 1          # not grounded in the original statement
                    continue
                cert = Certainty(it.certainty)
                # never upgrade: if rule-level cues show hedging/negation in the quoted clause, keep the weaker certainty
                clause = next((text[a:b] for a, b in _clauses(text) if a <= pos < b), text)
                if cert == Certainty.DEFINITE and _has_cue(clause, HEDGE_CUES):
                    cert = Certainty.HEDGED
                canon = map_term(it.name, table)
                obs.append(Observation(
                    category=cat, label=canon or normalize_phrase(it.name), canonical=canon, raw_text=it.name,
                    attribute=canon_color(it.color) if cat == Category.CLOTHING else None,
                    provenance=Provenance(source_type="narrative", source_id=source_id, prompt_version=self.prompt_id,
                                          evidence_ref=f"{narrative_id}:{pos}-{pos + len(anchor)}"),
                    uncertainty=UncertaintyInfo(linguistic_certainty=cert, needs_review=cert != Certainty.DEFINITE)))
        if parsed.scene.place:
            canon = map_term(parsed.scene.place, "scene")
            place = parsed.scene.place.lower().strip()
            pos = low.find(place) if place else -1
            anchor_len = len(place)
            if pos < 0 and canon:
                pos, anchor_len = low.find(canon), len(canon)
            if pos >= 0:
                # character span of the grounding text (provenance), as for the other categories
                obs.append(Observation(category=Category.SCENE, label=canon or normalize_phrase(parsed.scene.place),
                                       canonical=canon, raw_text=parsed.scene.place,
                                       provenance=Provenance(source_type="narrative", source_id=source_id,
                                                             prompt_version=self.prompt_id,
                                                             evidence_ref=f"{narrative_id}:{pos}-{pos + anchor_len}"),
                                       uncertainty=UncertaintyInfo(linguistic_certainty=Certainty.DEFINITE)))
            else:
                dropped += 1
        return NarrativeResult(narrative_id, text, obs, _missing(text, obs), unsupported_dropped=dropped, parse_status=status)
