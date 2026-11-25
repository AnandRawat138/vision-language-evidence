"""Shared post-processing: raw run outputs -> observations -> scored, evaluable records.

Caches (results/processed/cache/):
  quality.json             image-quality indicators per uid (original images)
  clip_<variant>.json      CLIP grounding scores per (image variant, label)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from vle.datasets.manifest import load_manifest
from vle.datasets.reference import load_reference
from vle.evidence import observations_from_vlm
from vle.inference.base import InferenceResult
from vle.lexicon import find_terms
from vle.parsing import parse_naive, parse_structured
from vle.preprocessing.quality import assess_quality, load_rgb
from vle.runio import ROOT, read_jsonl
from vle.schemas import Category, ImageQuality, Observation
from vle.uncertainty.calibration import VERBAL_MAP

CACHE = ROOT / "results" / "processed" / "cache"
OBJ_SOURCES = {"coco", "vqa"}          # exhaustive COCO-80 object references
ATTR_SOURCES = {"vg", "flickr30k"}     # clothing/accessory references (non-exhaustive)


def manifest_index() -> dict[str, dict]:
    return {r["uid"]: r for r in load_manifest(ROOT)}


def references(uids: list[str]) -> dict[str, dict]:
    return {u: load_reference(ROOT, u) for u in uids}


def quality_cache(uids: list[str]) -> dict[str, ImageQuality]:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "quality.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    man = manifest_index()
    changed = False
    for u in uids:
        if u not in data:
            data[u] = assess_quality(load_rgb(ROOT / man[u]["local_path"])).model_dump()
            changed = True
    if changed:
        p.write_text(json.dumps(data))
    return {u: ImageQuality(**data[u]) for u in uids}


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    cfg = json.loads((run_dir / "config.json").read_text())
    return cfg, read_jsonl(run_dir / "outputs.jsonl")


def run_observations(run_dir: Path, quality: dict[str, ImageQuality] | None = None,
                     key: str = "uid") -> tuple[dict[str, list[Observation]], dict[str, dict]]:
    """Returns ({key: observations}, {key: info}) where info has parse status, latency, tokens."""
    cfg, rows = load_run(run_dir)
    binfo = cfg["backend"]
    run_id = run_dir.name
    obs: dict[str, list[Observation]] = {}
    info: dict[str, dict] = {}
    for r in rows:
        k = r["image_ids"][0] if key == "image_id" else r["meta"].get("uid", r["image_ids"][0])
        res = InferenceResult(request_id=r["request_id"], text=r["text"], tokens=r.get("tokens", []),
                              token_logprobs=r.get("token_logprobs", []))
        parsed, status, err = parse_structured(r["text"]) if r.get("error") is None else (None, "failed", r["error"])
        naive, nstatus = parse_naive(r["text"])
        q = quality.get(r["meta"].get("uid")) if quality else None
        obs[k] = observations_from_vlm(parsed, res, k, binfo, r["prompt_id"], run_id, q) if parsed else []
        info[k] = {"parse_status": status, "naive_status": nstatus, "error": r.get("error") or err,
                   "n_output_tokens": r.get("n_output_tokens"), "latency_s": r.get("latency_s"),
                   "batch_size": r.get("batch_size"), "text": r["text"], "unclear": getattr(parsed, "unclear", []) if parsed else [],
                   "vlm_quality": parsed.image_quality.model_dump() if parsed else None}
    return obs, info


def caption_label_sets(text: str) -> dict[str, set]:
    """Baseline conversion of a free-form caption into label sets (same lexicon as the framework)."""
    return {"object": find_terms(text, "coco"), "clothing": find_terms(text, "clothing"),
            "accessory": find_terms(text, "accessories"), "carried_object": find_terms(text, "carried"),
            "scene": find_terms(text, "scene")}


# --------------------------------------------------------------- CLIP -----
def add_grounding(obs: dict[str, list[Observation]], images: dict[str, object], cache_name: str,
                  scorer=None) -> None:  # type: ignore[no-untyped-def]
    """Fill uncertainty.grounding_score using CLIP (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"clip_{cache_name}.json"
    cache: dict[str, dict[str, float]] = json.loads(p.read_text()) if p.exists() else {}
    need = {k: sorted({o.canonical for o in v if o.canonical and o.canonical not in cache.get(k, {})})
            for k, v in obs.items()}
    need = {k: v for k, v in need.items() if v}
    if need:
        if scorer is None:
            from vle.uncertainty.grounding import ClipScorer
            scorer = ClipScorer()
        for k, labels in need.items():
            img = images[k]() if callable(images[k]) else images[k]
            sc = scorer.scores(img, labels)
            cache.setdefault(k, {}).update({l: round(s, 5) for l, s in zip(labels, sc)})
        p.write_text(json.dumps(cache))
    for k, v in obs.items():
        for o in v:
            if o.canonical:
                o.uncertainty.grounding_score = cache.get(k, {}).get(o.canonical)


def feature_row(o: Observation) -> dict:
    u = o.uncertainty
    return {"token": u.token_confidence, "verbal": VERBAL_MAP.get(u.verbal_confidence, 0.6),
            "ground": u.grounding_score, "n_quality_flags": len(u.image_quality_flags),
            "is_object": int(o.category == Category.OBJECT), "is_clothing": int(o.category == Category.CLOTHING)}


def labelled_rows(obs: dict[str, list[Observation]], refs: dict[str, dict], uid_of=lambda k: k) -> list[dict]:
    """One row per (image, category, canonical) with correctness y where the reference supports it."""
    rows = []
    for k, v in obs.items():
        ref = refs[uid_of(k)]
        seen = set()
        for o in v:
            if o.canonical is None:
                continue
            key = (o.category, o.canonical)
            if key in seen:
                continue
            seen.add(key)
            y = None
            if o.category == Category.OBJECT and ref["objects_complete"]:
                y = int(o.canonical in ref["objects"])
            elif o.category == Category.CLOTHING and ref.get("attr_available"):
                y = int(o.canonical in ref["clothing"])
            if y is None:
                continue
            rows.append({"key": k, "uid": uid_of(k), "category": o.category.value, "label": o.canonical,
                         "y": y, "obs": o, **feature_row(o)})
    return rows


def pred_sets(obs: dict[str, list[Observation]], keep=lambda o: True) -> dict[str, dict[str, set]]:
    out: dict[str, dict[str, set]] = {}
    for k, v in obs.items():
        d: dict[str, set] = defaultdict(set)
        cols: dict[str, set] = defaultdict(set)
        for o in v:
            if o.canonical and keep(o):
                d[o.category.value].add(o.canonical)
                if o.category == Category.CLOTHING and o.attribute:
                    cols[o.canonical].add(o.attribute)
        d["clothing_colors"] = dict(cols)  # type: ignore[assignment]
        out[k] = d
    return out


def evaluate_predictions(preds: dict[str, dict[str, set]], refs: dict[str, dict], uids: list[str]) -> dict:
    """All Exp-1 metrics for a set of predictions on the given uids."""
    from vle.evaluation.metrics import chair, color_accuracy, evaluate_sets
    ou = [u for u in uids if refs[u]["objects_complete"]]
    au = [u for u in uids if refs[u].get("attr_available")]
    su = [u for u in uids if refs[u]["scene_places"]]
    out = {}
    out["objects"] = evaluate_sets({u: preds.get(u, {}).get("object", set()) for u in ou}, {u: set(refs[u]["objects"]) for u in ou})
    out["chair"] = chair({u: preds.get(u, {}).get("object", set()) for u in ou}, {u: set(refs[u]["objects"]) for u in ou})
    out["clothing"] = evaluate_sets({u: preds.get(u, {}).get("clothing", set()) for u in au}, {u: set(refs[u]["clothing"]) for u in au})
    out["accessory"] = evaluate_sets({u: preds.get(u, {}).get("accessory", set()) for u in au}, {u: set(refs[u]["accessories"]) for u in au})
    out["carried"] = evaluate_sets({u: preds.get(u, {}).get("carried_object", set()) for u in au}, {u: set(refs[u]["carried"]) for u in au})
    out["color"] = color_accuracy({u: preds.get(u, {}).get("clothing_colors", {}) for u in au}, {u: refs[u]["clothing"] for u in au})
    # scene: hit rate = predicted place canonical among caption-consensus places
    hits = [int(bool(preds.get(u, {}).get("scene", set()) & set(refs[u]["scene_places"]))) for u in su]
    out["scene"] = {"hit_rate": float(np.mean(hits)) if hits else float("nan"), "n_images": len(su), "per_image": hits}
    # attribute accuracy: per clothing reference item, detected (recall) — reported as attribute recall
    return out


def strip_per_image(m: dict) -> dict:
    """Drop bulky per-image vectors for summary JSON."""
    out = {}
    for k, v in m.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if kk not in ("per_image", "per_label_f1")}
        else:
            out[k] = v
    return out
