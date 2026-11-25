"""Convert validated VLM outputs into provenance-carrying Observation records."""

from __future__ import annotations

from vle.inference.base import InferenceResult
from vle.lexicon import canon_color, map_term, normalize_phrase
from vle.parsing import find_value_span
from vle.schemas import (Category, ImageQuality, Observation, Provenance, UncertaintyInfo,
                         VLMStructuredOutput)

_CAT_TABLE = {
    Category.OBJECT: ("objects", "coco"),
    Category.CLOTHING: ("clothing", "clothing"),
    Category.ACCESSORY: ("accessories", "accessories"),
    Category.CARRIED: ("carried_objects", "carried"),
}


def observations_from_vlm(parsed: VLMStructuredOutput, result: InferenceResult, image_id: str,
                          backend_info: dict, prompt_id: str, run_id: str,
                          quality: ImageQuality | None = None, evidence_ref: str | None = None) -> list[Observation]:
    obs: list[Observation] = []
    raw = result.text
    cursor = 0
    qflags = list(quality.flags) if quality else []
    for cat, (field, table) in _CAT_TABLE.items():
        for item in getattr(parsed, field):
            span = find_value_span(raw, item.name, cursor)
            tok_conf = result.span_confidence(*span) if span else None
            if span:
                cursor = span[1]
            canon = map_term(item.name, table)
            obs.append(Observation(
                category=cat, label=canon or normalize_phrase(item.name), raw_text=item.name,
                attribute=canon_color(item.color) if cat == Category.CLOTHING else None,
                canonical=canon,
                provenance=Provenance(source_type="vlm", source_id=image_id, image_id=image_id,
                                      model_name=backend_info.get("model_id"),
                                      model_revision=backend_info.get("model_revision"),
                                      quantization=backend_info.get("quantization"),
                                      prompt_version=prompt_id, run_id=run_id,
                                      evidence_ref=evidence_ref or image_id),
                uncertainty=UncertaintyInfo(verbal_confidence=item.confidence, token_confidence=tok_conf,
                                            image_quality_flags=qflags)))
    sc = parsed.scene
    if sc.place:
        canon = map_term(sc.place, "scene")
        span = find_value_span(raw, sc.place)
        obs.append(Observation(
            category=Category.SCENE, label=canon or normalize_phrase(sc.place), raw_text=sc.place,
            canonical=canon, attribute=sc.setting,
            provenance=Provenance(source_type="vlm", source_id=image_id, image_id=image_id,
                                  model_name=backend_info.get("model_id"), model_revision=backend_info.get("model_revision"),
                                  quantization=backend_info.get("quantization"), prompt_version=prompt_id, run_id=run_id,
                                  evidence_ref=evidence_ref or image_id),
            uncertainty=UncertaintyInfo(token_confidence=result.span_confidence(*span) if span else None,
                                        image_quality_flags=qflags)))
    return obs


def label_sets(obs: list[Observation], min_score: float | None = None, score_field: str = "combined_score") -> dict[str, set]:
    """Canonical label sets per category, optionally keeping only observations with score >= min_score."""
    out: dict[str, set] = {"object": set(), "clothing": set(), "accessory": set(), "carried_object": set(), "scene": set()}
    colors: dict[str, set] = {}
    for o in obs:
        if o.canonical is None:
            continue
        if min_score is not None:
            s = getattr(o.uncertainty, score_field)
            if s is not None and s < min_score:
                continue
        out[o.category.value].add(o.canonical)
        if o.category == Category.CLOTHING and o.attribute:
            colors.setdefault(o.canonical, set()).add(o.attribute)
    out["clothing_colors"] = colors  # type: ignore[assignment]
    return out
