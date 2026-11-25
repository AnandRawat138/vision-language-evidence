"""Generic VLM inference runner used by Experiments 1, 3, 5 and baseline B3.

Examples
    # framework extraction on all 1,000 images (static batching, B=8)
    python experiments/run_inference.py --exp exp1 --name struct_b8 --prompt struct-v1 --policy static --batch 8
    # degraded copies for robustness
    python experiments/run_inference.py --exp exp5 --name blur_3 --prompt struct-v1 --degrade blur:3 --subset robust
    # multi-view groups, independent per view / contextual
    python experiments/run_inference.py --exp exp3 --name views_indep --mode views
    python experiments/run_inference.py --exp exp3 --name views_context --mode context

Request ids are deterministic (<uid>|<prompt>|<variant>) so interrupted runs resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from vle.batching.scheduler import BatchScheduler, SchedulerConfig  # noqa: E402
from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.inference import make_backend  # noqa: E402
from vle.inference.base import InferenceRequest  # noqa: E402
from vle.preprocessing.quality import crop_view, degrade, load_rgb  # noqa: E402
from vle.prompts import get_prompt, render  # noqa: E402
from vle.runio import environment, run_dir, write_json  # noqa: E402

VIEW_SPECS = [("left", None), ("right", None), ("full", "lowlight:2.2"), ("full", "blur:2.5")]


def subset_rows(name: str, seed: int = 2026) -> list[dict]:
    rows = load_manifest(ROOT)
    if name == "all":
        return rows
    if name in ("dev", "test"):
        return [r for r in rows if r["split"] == name]
    if name == "robust":
        # 200 test images with exhaustive object references (coco+vqa), fixed seed
        # 120 test images with exhaustive object references (coco+vqa), fixed seed (time budget: see docs/PROGRESS.md)
        cand = [r for r in rows if r["split"] == "test" and r["source"] in ("coco", "vqa")]
        return random.Random(seed).sample(cand, 120)
    if name == "robust_attr":
        cand = [r for r in rows if r["split"] == "test" and r["source"] in ("vg", "flickr30k")]
        return random.Random(seed).sample(cand, 100)
    if name == "groups_b1":
        return subset_rows("groups_test", seed)[:30]
    if name == "groups_all":
        return subset_rows("groups_dev", seed) + subset_rows("groups_test", seed)
    if name.startswith("groups"):
        # coco/vqa images for multi-view groups (dev: 40 for tuning, test: 120)
        split = "dev" if name.endswith("dev") else "test"
        cand = [r for r in rows if r["split"] == split and r["source"] in ("coco", "vqa")]
        return random.Random(seed + 1).sample(cand, 30 if split == "dev" else 90)
    if name.startswith("bench"):
        n = int(name.replace("bench", "") or 64)
        cand = [r for r in rows if r["split"] == "dev"]
        return random.Random(seed + 2).sample(cand, n)
    raise ValueError(name)


def make_views(img, spec):  # type: ignore[no-untyped-def]
    which, deg = spec
    if which in ("left", "right", "top", "bottom"):
        img, _ = crop_view(img, which)
    if deg:
        k, lv = deg.split(":")
        img = degrade(img, k, float(lv), seed=0)
    return img


def build_requests(args, rows) -> list[InferenceRequest]:  # type: ignore[no-untyped-def]
    p = get_prompt(args.prompt)
    reqs = []
    for r in rows:
        img = load_rgb(ROOT / r["local_path"])
        if args.degrade:
            k, lv = args.degrade.split(":")
            img = degrade(img, k, float(lv), seed=int(r["md5"][:8], 16) % 10_000)
        if args.mode == "single":
            reqs.append(InferenceRequest(prompt_id=args.prompt, text=render(args.prompt), system=p.get("system"),
                                         images=[img], image_ids=[r["uid"]], max_new_tokens=p["max_new_tokens"],
                                         request_id=f"{r['uid']}|{args.prompt}|{args.degrade or 'orig'}",
                                         meta={"uid": r["uid"], "degrade": args.degrade}))
        else:
            views = [make_views(img, s) for s in VIEW_SPECS]
            if args.mode == "views":
                for vi, v in enumerate(views):
                    reqs.append(InferenceRequest(prompt_id=args.prompt, text=render(args.prompt), system=p.get("system"),
                                                 images=[v], image_ids=[f"{r['uid']}#v{vi}"], max_new_tokens=p["max_new_tokens"],
                                                 request_id=f"{r['uid']}#v{vi}|{args.prompt}",
                                                 meta={"uid": r["uid"], "view": vi, "view_spec": VIEW_SPECS[vi]}))
            else:  # context
                cp = get_prompt("context-v1")
                reqs.append(InferenceRequest(prompt_id="context-v1", text=render("context-v1", n=len(views)),
                                             system=cp.get("system"), images=views,
                                             image_ids=[f"{r['uid']}#v{i}" for i in range(len(views))],
                                             max_new_tokens=cp["max_new_tokens"], request_id=f"{r['uid']}|context-v1",
                                             meta={"uid": r["uid"], "views": VIEW_SPECS}))
    return reqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "local_gpu.yaml"))
    ap.add_argument("--prompt", default="struct-v1")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--mode", default="single", choices=["single", "views", "context"])
    ap.add_argument("--degrade", default=None)
    ap.add_argument("--policy", default="static")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = run_dir(args.exp, args.name)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    rows = subset_rows(args.subset)[: args.limit]
    reqs = build_requests(args, rows)
    if args.mode == "context":
        args.batch = min(args.batch, 2)
    backend = make_backend(cfg)
    sc = SchedulerConfig(policy=args.policy, batch_size=args.batch, checkpoint_path=str(out / "outputs.jsonl"),
                         timeout_s=cfg.get("timeout_s", 900))
    sched = BatchScheduler(backend, sc)
    write_json(out / "config.json", {"args": vars(args), "backend": backend.info(), "scheduler": sc.__dict__,
                                     "prompt": get_prompt(args.prompt) if args.mode != "context" else get_prompt("context-v1"),
                                     "environment": environment(), "n_requests": len(reqs),
                                     "uids": [r["uid"] for r in rows]})
    stats = sched.run(reqs)
    sched.close()
    write_json(out / f"stats_{stats.n_requests}.json", {**stats.summary(), "batch_sizes": stats.batch_sizes,
                                                        "batch_latencies": stats.batch_latencies})
    logging.info("done: %s", json.dumps(stats.summary()))


if __name__ == "__main__":
    main()
