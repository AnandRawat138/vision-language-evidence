"""Derive evaluation references from *original* dataset annotations.

Output: data/annotations/reference/<uid>.json with fields

  objects          COCO categories present (exhaustive for coco/vqa sources)
  objects_complete True when the object list is exhaustive w.r.t. the 80 COCO categories
  object_boxes     [{category, bbox[x,y,w,h], area}]   (coco/vqa only)
  clothing         {canonical: [colours]}    (vg: VG object+attribute annotations;
                                              flickr30k: Flickr30K-Entities 'clothing' phrases)
  accessories      [canonical]
  carried          [canonical]
  attr_complete    False everywhere: VG/Entities do not exhaustively annotate clothing,
                   so precision on these categories is a lower bound.
  scene_places     canonical places mentioned by >=2 independent captions
  captions         original human captions (coco, vqa, flickr30k)
  flickr_phrases   typed phrase annotations (flickr30k)
  vqa              VQA v2 question/answer annotations (vqa)

The derivation rules are deterministic and documented in docs/annotation_schema.md.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from vle.lexicon import LEXICON_VERSION, canon_color, find_terms, map_term

REFERENCE_VERSION = "ref-v1"
SCENE_MIN_CAPTIONS = 2


def scene_from_captions(captions: list[str]) -> list[str]:
    c: Counter = Counter()
    for cap in captions:
        for s in find_terms(cap, "scene"):
            c[s] += 1
    return sorted(s for s, k in c.items() if k >= SCENE_MIN_CAPTIONS)


def coco_reference(image_id: int, coco: dict) -> dict:
    anns = [a for a in coco["instances"].get(image_id, [])]
    caps = coco["captions"].get(image_id, [])
    return {
        "objects": sorted({a["category"] for a in anns}),
        "objects_complete": True,
        "object_counts": dict(Counter(a["category"] for a in anns)),
        "object_boxes": [{"category": a["category"], "bbox": a["bbox"], "area": a["area"]} for a in anns],
        "clothing": {}, "accessories": [], "carried": [],
        "attr_available": False,
        "captions": caps,
        "scene_places": scene_from_captions(caps),
    }


def vg_reference(objs: list[dict]) -> dict:
    clothing: dict[str, set] = defaultdict(set)
    acc, carried, objects = set(), set(), set()
    for o in objs:
        attrs = o.get("attributes", []) or []
        for nm in o.get("names", []):
            c = map_term(nm, "clothing")
            if c:
                for a in attrs:
                    col = canon_color(a)
                    if col:
                        clothing[c].add(col)
                clothing.setdefault(c, set())
            a_ = map_term(nm, "accessories")
            if a_:
                acc.add(a_)
            cr = map_term(nm, "carried")
            if cr and cr not in {"food", "book", "sports equipment"}:
                carried.add(cr)
            ob = map_term(nm, "coco")
            if ob:
                objects.add(ob)
    return {
        "objects": sorted(objects),
        "objects_complete": False,
        "object_boxes": [],
        "clothing": {k: sorted(v) for k, v in clothing.items()},
        "accessories": sorted(acc), "carried": sorted(carried),
        "attr_available": True,
        "captions": [],
        "scene_places": [],
    }


def flickr_reference(ent: dict) -> dict:
    clothing: dict[str, set] = defaultdict(set)
    acc, carried, objects = set(), set(), set()
    caps = [s["sentence"] for s in ent["sentences"]]
    phrases = []
    for s in ent["sentences"]:
        for p in s["phrases"]:
            phrases.append(p)
            text = p["phrase"]
            if "clothing" in p["types"]:
                c = map_term(text, "clothing")
                if c:
                    col = canon_color(text)
                    clothing[c]
                    if col:
                        clothing[c].add(col)
                a_ = map_term(text, "accessories")
                if a_:
                    acc.add(a_)
            if "other" in p["types"] or "instruments" in p["types"]:
                cr = map_term(text, "carried")
                if cr and cr not in {"food", "book", "sports equipment"}:
                    carried.add(cr)
            ob = map_term(text, "coco")
            if ob and not {"bodyparts", "scene"} & set(p["types"]):
                objects.add(ob)
    return {
        "objects": sorted(objects),
        "objects_complete": False,
        "object_boxes": [],
        "clothing": {k: sorted(v) for k, v in clothing.items()},
        "accessories": sorted(acc), "carried": sorted(carried),
        "attr_available": True,
        "captions": caps,
        "flickr_phrases": phrases,
        "scene_places": scene_from_captions(caps),
    }


def write_reference(uid: str, ref: dict, source: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = {"uid": uid, "source": source, "reference_version": REFERENCE_VERSION,
           "lexicon_version": LEXICON_VERSION, "provenance": "derived from original dataset annotations", **ref}
    (out_dir / f"{uid}.json").write_text(json.dumps(ref, indent=1), encoding="utf-8")


def load_reference(project_root: Path, uid: str) -> dict:
    return json.loads((project_root / "data" / "annotations" / "reference" / f"{uid}.json").read_text(encoding="utf-8"))
