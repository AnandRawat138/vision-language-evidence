"""Prompt selection on DEV images only (no test labels): compares structured prompts on
length-truncation rate, parse status, output length and dev object F1.

    python experiments/prompt_dev_check.py --prompts struct-v1 struct-v2 --n 48
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.datasets.reference import load_reference  # noqa: E402
from vle.evaluation.metrics import prf  # noqa: E402
from vle.inference import make_backend  # noqa: E402
from vle.inference.base import InferenceRequest  # noqa: E402
from vle.lexicon import map_term  # noqa: E402
from vle.parsing import parse_structured  # noqa: E402
from vle.preprocessing.quality import load_rgb  # noqa: E402
from vle.prompts import get_prompt, render  # noqa: E402
from vle.runio import environment, run_dir, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", nargs="+", default=["struct-v1", "struct-v2"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    rows = [r for r in load_manifest(ROOT, split="dev") if r["source"] in ("coco", "vqa")]
    rows = random.Random(11).sample(rows, args.n)
    be = make_backend(yaml.safe_load((ROOT / "configs" / "local_gpu.yaml").read_text()))
    out = run_dir("prompt_dev", "check")
    summary = {"environment": environment(), "backend": be.info(), "uids": [r["uid"] for r in rows], "prompts": {}}
    for pid in args.prompts:
        p = get_prompt(pid)
        recs = []
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            reqs = [InferenceRequest(prompt_id=pid, text=render(pid), system=p["system"], images=[load_rgb(ROOT / r["local_path"])],
                                     image_ids=[r["uid"]], max_new_tokens=p["max_new_tokens"]) for r in chunk]
            for r, res in zip(chunk, be.generate_batch(reqs)):
                recs.append((r["uid"], res))
        T = F = N = 0
        trunc = 0
        status = {}
        toks = []
        for uid, res in recs:
            parsed, st, _ = parse_structured(res.text)
            status[st] = status.get(st, 0) + 1
            trunc += int(res.n_output_tokens >= p["max_new_tokens"])
            toks.append(res.n_output_tokens)
            pred = {map_term(o.name, "coco") for o in parsed.objects} - {None} if parsed else set()
            g = set(load_reference(ROOT, uid)["objects"])
            T, F, N = T + len(pred & g), F + len(pred - g), N + len(g - pred)
        summary["prompts"][pid] = {"truncated_rate": trunc / len(recs), "parse": status, "mean_out_tokens": sum(toks) / len(toks),
                                   "dev_object_p_r_f1": prf(T, F, N), "sha1": p["sha1"],
                                   "wall_s_total": sum(r.latency_s for _, r in recs[:: args.batch])}
        print(pid, json.dumps(summary["prompts"][pid]))
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
