"""Milestone 1/4: local-GPU feasibility probe for a candidate VLM.

    python scripts/smoke_gpu.py --model Qwen/Qwen2.5-VL-3B-Instruct --quant nf4 --batches 1 4 8

Writes results/raw/feasibility/<model>_<quant>.json (load time, VRAM, latency, sample output).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.inference.base import InferenceRequest, OutOfMemory  # noqa: E402
from vle.inference.hf_backend import HFVisionLanguageBackend  # noqa: E402
from vle.parsing import parse_structured  # noqa: E402
from vle.preprocessing.quality import load_rgb  # noqa: E402
from vle.prompts import get_prompt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--quant", default="nf4")
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--n-images", type=int, default=8)
    args = ap.parse_args()
    rows = load_manifest(ROOT, split="dev")[: max(args.batches + [args.n_images])]
    imgs = [load_rgb(ROOT / r["local_path"]) for r in rows]
    torch.cuda.reset_peak_memory_stats()
    be = HFVisionLanguageBackend(model_id=args.model, quantization=args.quant, max_pixels=args.max_pixels)
    after_load = torch.cuda.memory_allocated() / 2**20
    p = get_prompt("struct-v1")
    out = {"info": be.info(), "weights_mb": round(after_load, 1), "runs": []}
    for b in args.batches:
        reqs = [InferenceRequest(prompt_id="struct-v1", text=p["text"], system=p["system"], images=[imgs[i % len(imgs)]],
                                 image_ids=[rows[i % len(rows)]["uid"]], max_new_tokens=p["max_new_tokens"]) for i in range(b)]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            res = be.generate_batch(reqs)
        except OutOfMemory as e:
            out["runs"].append({"batch": b, "error": str(e)})
            print(b, "OOM")
            continue
        dt = time.perf_counter() - t0
        parsed = [parse_structured(r.text)[1] for r in res]
        run = {"batch": b, "wall_s": round(dt, 2), "img_per_s": round(b / dt, 3),
               "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
               "out_tokens_mean": sum(r.n_output_tokens for r in res) / b,
               "in_tokens_mean": sum(r.n_input_tokens for r in res) / b, "parse": parsed,
               "sample": res[0].text[:600]}
        out["runs"].append(run)
        print(json.dumps({k: v for k, v in run.items() if k != "sample"}))
    print(out["runs"][0].get("sample"))
    d = ROOT / "results" / "raw" / "feasibility"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.model.replace('/', '_')}_{args.quant}_{args.max_pixels}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
