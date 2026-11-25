"""Dataset statistics table (Table 1) generated from the manifest and the derived references."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.datasets.reference import load_reference  # noqa: E402
from vle.reporting.tables import write_table  # noqa: E402
from vle.runio import write_json  # noqa: E402


def main() -> None:
    rows = load_manifest(ROOT)
    agg: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ref = load_reference(ROOT, r["uid"])
        a = agg[r["source"]]
        a["split_" + r["split"]].append(1)
        a["objects"].append(len(ref["objects"]))
        a["clothing"].append(len(ref["clothing"]))
        a["acc_carr"].append(len(ref["accessories"]) + len(ref["carried"]))
        a["scene"].append(int(bool(ref["scene_places"])))
        a["pixels"].append(int(r["width"]) * int(r["height"]) / 1e6)
        a["captions"].append(len(ref["captions"]))
        a["vqa"].append(len(ref.get("vqa", [])))
    out = []
    stats = {}
    names = {"coco": "COCO 2017 val", "vqa": "VQA v2 val", "vg": "Visual Genome 1.2", "flickr30k": "Flickr30K (Entities test)"}
    refs = {"coco": "objects$^*$, captions", "vqa": "objects$^*$, captions, QA", "vg": "objects, garments+colour",
            "flickr30k": "phrases, garments+colour"}
    for s in ("coco", "vqa", "vg", "flickr30k"):
        a = agg[s]
        d = {"n": len(a["objects"]), "dev": len(a["split_dev"]), "test": len(a["split_test"]),
             "objects_mean": float(np.mean(a["objects"])), "garments_mean": float(np.mean(a["clothing"])),
             "acc_carried_mean": float(np.mean(a["acc_carr"])), "scene_frac": float(np.mean(a["scene"])),
             "megapixels_mean": float(np.mean(a["pixels"]))}
        stats[s] = d
        out.append([names[s], f"{d['n']} ({d['dev']}/{d['test']})", f"{d['objects_mean']:.2f}", f"{d['garments_mean']:.2f}",
                    f"{d['acc_carried_mean']:.2f}", f"{d['scene_frac']:.2f}", f"{d['megapixels_mean']:.2f}", refs[s]])
    write_json(ROOT / "results" / "processed" / "dataset_stats.json", stats)
    write_table("tab_data", ["Source", "Images (dev/test)", "Obj./img", "Garm./img", "Acc.+carr./img", "Scene ref.", "MP", "Reference"],
                out, "Benchmark composition (1,000 images after de-duplication). Per-image means of reference labels derived from "
                "original annotations; $^*$exhaustive over the 80 COCO classes. MP: mean megapixels.", "tab:data",
                colspec="lrrrrrrl", size=r"\scriptsize")
    print(stats)


if __name__ == "__main__":
    main()
