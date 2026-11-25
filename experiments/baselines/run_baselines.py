"""Baselines B1 (BLIP conventional captioning) and B2 (CLIP zero-shot multi-label) on all 1,000 images.

    python experiments/baselines/run_baselines.py

Outputs: results/raw/baselines/blip_captions.jsonl, clip_scores.json (+ timing).
Label thresholds for CLIP are selected on the DEV split only (see analyze_exp1.py).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.lexicon import ACCESSORIES, CARRIED, CLOTHING, COCO_CATEGORIES  # noqa: E402
from vle.preprocessing.quality import load_rgb  # noqa: E402
from vle.runio import environment, run_dir, write_json  # noqa: E402

BLIP_ID = "Salesforce/blip-image-captioning-base"
CLIP_ID = "openai/clip-vit-base-patch32"


def clip_label_space() -> dict[str, list[str]]:
    return {"object": COCO_CATEGORIES, "clothing": list(CLOTHING), "accessory": list(ACCESSORIES),
            "carried_object": [c for c in CARRIED if c not in ("food", "book", "sports equipment")]}


@torch.inference_mode()
def main() -> None:
    rows = load_manifest(ROOT)
    out = run_dir("baselines", "b1_b2")
    env = environment()
    # ---------------- B1: BLIP
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(BLIP_ID)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_ID, torch_dtype=torch.float16).cuda().eval()
    t0 = time.perf_counter()
    caps = []
    B = 16
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        imgs = [load_rgb(ROOT / r["local_path"]) for r in chunk]
        inp = proc(images=imgs, return_tensors="pt").to("cuda", torch.float16)
        ids = model.generate(**inp, max_new_tokens=40, num_beams=3)
        for r, c in zip(chunk, proc.batch_decode(ids, skip_special_tokens=True)):
            caps.append({"uid": r["uid"], "caption": c.strip()})
    blip_t = time.perf_counter() - t0
    (out / "blip_captions.jsonl").write_text("\n".join(json.dumps(c) for c in caps), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    # ---------------- B2: CLIP
    from vle.uncertainty.grounding import ClipScorer
    cs = ClipScorer(CLIP_ID)
    space = clip_label_space()
    labels = sorted({l for v in space.values() for l in v})
    te = cs.text_emb(labels)
    t0 = time.perf_counter()
    scores = {}
    for i in range(0, len(rows), 32):
        chunk = rows[i:i + 32]
        ie = cs.image_emb([load_rgb(ROOT / r["local_path"]) for r in chunk])
        s = (ie @ te.T).cpu().numpy()
        for r, v in zip(chunk, s):
            scores[r["uid"]] = {l: round(float(x), 5) for l, x in zip(labels, v)}
    clip_t = time.perf_counter() - t0
    write_json(out / "clip_scores.json", {"label_space": space, "scores": scores})
    write_json(out / "config.json", {"blip": {"model": BLIP_ID, "decoding": "beam=3, max_new_tokens=40", "fp16": True,
                                              "batch": B, "wall_s": blip_t},
                                     "clip": {"model": CLIP_ID, "prompt": "a photo of a {}.", "wall_s": clip_t},
                                     "environment": env, "n_images": len(rows)})
    print("BLIP", blip_t, "CLIP", clip_t)


if __name__ == "__main__":
    main()
