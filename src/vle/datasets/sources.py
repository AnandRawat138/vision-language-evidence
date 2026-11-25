"""Readers for the original public-dataset annotations.

Each reader returns plain dictionaries in the dataset's *original* terms;
conversion into the evaluation reference format happens in ``reference.py``.
Nothing here writes model outputs or new annotations.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path


def _load_json_from_zip(zpath: Path, member: str) -> object:
    with zipfile.ZipFile(zpath) as z:
        with z.open(member) as f:
            return json.load(f)


# ---------------------------------------------------------------- COCO -----
def load_coco(raw_dir: Path, split: str) -> dict:
    """Return {'images': {id: info}, 'instances': {id: [ann]}, 'captions': {id: [str]}, 'cats': {id: name}}."""
    z = raw_dir / "annotations_trainval2017.zip"
    inst = _load_json_from_zip(z, f"annotations/instances_{split}.json")
    caps = _load_json_from_zip(z, f"annotations/captions_{split}.json")
    cats = {c["id"]: c["name"] for c in inst["categories"]}  # type: ignore[index]
    images = {im["id"]: im for im in inst["images"]}  # type: ignore[index]
    instances: dict[int, list] = defaultdict(list)
    for a in inst["annotations"]:  # type: ignore[index]
        instances[a["image_id"]].append({"category": cats[a["category_id"]], "bbox": a["bbox"],
                                         "area": a["area"], "iscrowd": a["iscrowd"]})
    captions: dict[int, list] = defaultdict(list)
    for a in caps["annotations"]:  # type: ignore[index]
        captions[a["image_id"]].append(a["caption"].strip())
    licenses = {l["id"]: l for l in inst["licenses"]}  # type: ignore[index]
    return {"images": images, "instances": dict(instances), "captions": dict(captions),
            "cats": cats, "licenses": licenses}


# ----------------------------------------------------------------- VQA -----
def load_vqa_val(raw_dir: Path) -> dict[int, list[dict]]:
    qs = _load_json_from_zip(raw_dir / "v2_Questions_Val_mscoco.zip", "v2_OpenEnded_mscoco_val2014_questions.json")
    ans = _load_json_from_zip(raw_dir / "v2_Annotations_Val_mscoco.zip", "v2_mscoco_val2014_annotations.json")
    qmap = {q["question_id"]: q for q in qs["questions"]}  # type: ignore[index]
    out: dict[int, list[dict]] = defaultdict(list)
    for a in ans["annotations"]:  # type: ignore[index]
        q = qmap[a["question_id"]]
        out[a["image_id"]].append({
            "question_id": a["question_id"], "question": q["question"],
            "answer_type": a["answer_type"], "question_type": a["question_type"],
            "multiple_choice_answer": a["multiple_choice_answer"],
            "answers": [x["answer"] for x in a["answers"]],
        })
    return dict(out)


# ------------------------------------------------------------------ VG -----
def load_vg_image_data(raw_dir: Path) -> list[dict]:
    return _load_json_from_zip(raw_dir / "image_data.json.zip", "image_data.json")  # type: ignore[return-value]


def iter_vg_attributes(raw_dir: Path):
    """Yield per-image dicts {'image_id', 'attributes': [ {names, attributes, synsets, x,y,w,h} ]}."""
    data = _load_json_from_zip(raw_dir / "attributes.json.zip", "attributes.json")
    yield from data  # type: ignore[misc]


# ------------------------------------------------------------ Flickr30K -----
_EN_RE = re.compile(r"\[/EN#(\d+)/([^\s]+) ([^\]]+)\]")


def parse_flickr_sentence(line: str) -> dict:
    phrases = []
    plain_parts: list[str] = []
    pos, plen = 0, 0
    for m in _EN_RE.finditer(line):
        before = line[pos:m.start()]
        plain_parts.append(before)
        plen += len(before)
        ph = m.group(3)
        phrases.append({"entity_id": m.group(1), "types": m.group(2).split("/"), "phrase": ph,
                        "start": plen, "end": plen + len(ph)})
        plain_parts.append(ph)
        plen += len(ph)
        pos = m.end()
    plain_parts.append(line[pos:])
    plain = "".join(plain_parts)
    # strip trailing whitespace only (leading strip would shift spans)
    return {"sentence": plain.rstrip(), "phrases": phrases}


def load_flickr_entities(raw_dir: Path, image_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with zipfile.ZipFile(raw_dir / "flickr30k_entities_annotations.zip") as z:
        names = set(z.namelist())
        for iid in image_ids:
            sent_name = next((n for n in (f"Sentences/{iid}.txt", f"annotations/Sentences/{iid}.txt") if n in names), None)
            xml_name = next((n for n in (f"Annotations/{iid}.xml", f"annotations/Annotations/{iid}.xml") if n in names), None)
            if sent_name is None:
                continue
            sents = [parse_flickr_sentence(l) for l in z.read(sent_name).decode("utf-8").splitlines() if l.strip()]
            boxes: dict[str, list] = defaultdict(list)
            size = None
            if xml_name:
                root = ET.fromstring(z.read(xml_name))
                s = root.find("size")
                if s is not None:
                    size = (int(s.findtext("width", "0")), int(s.findtext("height", "0")))
                for obj in root.findall("object"):
                    bb = obj.find("bndbox")
                    for nm in obj.findall("name"):
                        if bb is not None:
                            boxes[nm.text or ""].append([int(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")])
            out[iid] = {"sentences": sents, "boxes": dict(boxes), "size": size}
    return out
