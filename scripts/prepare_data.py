"""Milestone 3: build the 1,000-image benchmark manifest and derived references.

    python scripts/prepare_data.py

Deterministic given SEED and the dataset versions recorded in the manifest meta.
Images are downloaded from the official hosts into data/images/ (git-ignored,
not redistributed).
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vle.datasets import manifest as M  # noqa: E402
from vle.datasets import reference as R  # noqa: E402
from vle.datasets import sources as S  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "results" / "logs" / "prepare_data.log")])
log = logging.getLogger("prepare_data")


def main() -> None:
    raw = ROOT / "data" / "raw"
    rng = random.Random(M.SEED)

    log.info("loading COCO val2017")
    coco_val = S.load_coco(raw / "coco", "val2017")
    log.info("loading COCO train2017 (for VQA val2014 images)")
    coco_tr = S.load_coco(raw / "coco", "train2017")
    log.info("loading VQA v2 val")
    vqa = S.load_vqa_val(raw / "vqa")
    log.info("loading VG image data + attributes")
    vg_imgs = S.load_vg_image_data(raw / "vg")
    vg_attr = {}
    for d in S.iter_vg_attributes(raw / "vg"):
        vg_attr[d["image_id"]] = [{"names": o.get("names", []), "attributes": o.get("attributes", [])}
                                  for o in d.get("attributes", [])]
    gc.collect()
    flickr_test = (raw / "flickr30k" / "test.txt").read_text().split()

    coco_sel = M.select_coco(coco_val, M.ALLOCATION["coco"], rng)
    vqa_sel = M.select_vqa(vqa, coco_tr, {int(e.source_id) for e in coco_sel}, M.ALLOCATION["vqa"], rng)
    vg_sel = M.select_vg(vg_imgs, vg_attr, M.ALLOCATION["vg"], rng)
    fl_sel = M.select_flickr(flickr_test, M.ALLOCATION["flickr30k"], rng)
    # priority order for deduplication: coco > vqa > vg > flickr30k
    entries = coco_sel + vqa_sel + vg_sel + fl_sel
    log.info("candidates: %d", len(entries))

    M.download_images(entries, ROOT / "data" / "images")
    for e in entries:
        M.validate_and_hash(e, ROOT)
    M.deduplicate(entries)
    M.trim_and_split(entries, rng)

    # derived references (original annotations only)
    ref_dir = ROOT / "data" / "annotations" / "reference"
    fl_ids = [e.source_id for e in entries if e.source == "flickr30k" and e.status == "ok"]
    fl_ent = S.load_flickr_entities(raw / "flickr30k", fl_ids)
    for e in entries:
        if e.status != "ok":
            continue
        if e.source == "coco":
            ref = R.coco_reference(int(e.source_id), coco_val)
        elif e.source == "vqa":
            ref = R.coco_reference(int(e.source_id), coco_tr)
            ref["vqa"] = vqa[int(e.source_id)]
        elif e.source == "vg":
            ref = R.vg_reference(vg_attr[int(e.source_id)])
        else:
            if e.source_id not in fl_ent:
                e.status, e.exclusion_reason = "invalid", "no Entities annotation"
                continue
            ref = R.flickr_reference(fl_ent[e.source_id])
        R.write_reference(e.uid, ref, e.source, ref_dir)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": M.SEED, "allocation": M.ALLOCATION, "oversample": M.OVERSAMPLE, "dev_fraction": M.DEV_FRACTION,
        "python": platform.python_version(),
        "dataset_versions": {
            "coco": "COCO 2017 (val2017 images; annotations_trainval2017.zip)",
            "vqa": "VQA v2 val (v2_Questions_Val_mscoco, v2_Annotations_Val_mscoco); images = COCO val2014 present in train2017",
            "vg": "Visual Genome v1.2 image_data.json + attributes.json (v1.2)",
            "flickr30k": "Flickr30K images (HF mirror nlphuji/flickr30k zip) + Flickr30K Entities annotations, official test split",
        },
        "selection_rules": {
            "coco": "val2017 images with >=1 instance and captions; 60% containing person, 40% not; random (seed)",
            "vqa": ">=3 VQA v2 val questions; image file in COCO train2017; not in COCO subset",
            "vg": "VG images without coco_id that have >=1 object mapped to the clothing lexicon",
            "flickr30k": "random from Flickr30K Entities official test split",
            "dedup": "md5 exact + dHash(64-bit) Hamming<=4, priority coco>vqa>vg>flickr30k",
            "validation": "decodable by PIL, min side >= 64 px",
        },
    }
    M.write_manifest(entries, ROOT / "data" / "manifests", meta)
    from collections import Counter
    log.info("final: %s", Counter((e.source, e.split) for e in entries if e.status == "ok"))
    log.info("excluded: %s", Counter(e.status for e in entries if e.status != "ok"))


if __name__ == "__main__":
    main()
