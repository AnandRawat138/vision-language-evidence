"""Experiment 2: batch-processing scalability on the local GPU.

Offline: the same 64-image workload (dev split; labels unused) is processed with
  independent (B=1), static B in {4,8,16,32,64}, length-sorted static B=16, and adaptive,
  each repeated R times (default 3) after one warm-up batch.
Online: Poisson arrivals at fixed rates; independent vs dynamic batching (B_max, max_wait).

All raw per-request records and per-run stats are written under results/raw/exp2/.

    python experiments/batching/run_exp2.py --part offline --repeats 3
    python experiments/batching/run_exp2.py --part online
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import torch  # noqa: E402
import yaml  # noqa: E402

from run_inference import subset_rows  # noqa: E402
from vle.batching.scheduler import BatchScheduler, SchedulerConfig  # noqa: E402
from vle.inference import make_backend  # noqa: E402
from vle.inference.base import InferenceRequest  # noqa: E402
from vle.preprocessing.quality import load_rgb  # noqa: E402
from vle.prompts import get_prompt, render  # noqa: E402
from vle.runio import environment, run_dir, write_json  # noqa: E402

log = logging.getLogger("exp2")


def make_reqs(rows, tag: str) -> list[InferenceRequest]:  # type: ignore[no-untyped-def]
    p = get_prompt("struct-v1")
    return [InferenceRequest(prompt_id="struct-v1", text=render("struct-v1"), system=p["system"],
                             images=[load_rgb(ROOT / r["local_path"])], image_ids=[r["uid"]],
                             max_new_tokens=p["max_new_tokens"], request_id=f"{r['uid']}|{tag}",
                             meta={"uid": r["uid"]}) for r in rows]


def run_one(backend, rows, cfg: SchedulerConfig, out: Path, tag: str, online_rate: float | None = None) -> dict:  # type: ignore[no-untyped-def]
    reqs = make_reqs(rows, tag)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg.checkpoint_path = str(out / f"{tag}.jsonl")
    if Path(cfg.checkpoint_path).exists():
        Path(cfg.checkpoint_path).unlink()   # throughput runs are never resumed
    s = BatchScheduler(backend, cfg)
    st = s.run_online(reqs, online_rate, seed=0) if online_rate else s.run(reqs)
    s.close()
    summ = {**st.summary(), "tag": tag, "config": {k: v for k, v in cfg.__dict__.items() if k != "checkpoint_path"},
            "batch_sizes": st.batch_sizes, "batch_latencies": [round(x, 4) for x in st.batch_latencies],
            "adaptive_trace": st.adaptive_trace, "request_latencies": [round(x, 4) for x in st.request_latencies]}
    write_json(out / f"{tag}.stats.json", summ)
    log.info("%s: %.3f img/s p95=%.2fs gpu=%s ok=%d fail=%d oom=%d", tag, summ["throughput_img_s"] or 0,
             summ["latency_p95_s"], summ.get("gpu_mem_peak_mb"), st.n_ok, st.n_failed, st.n_oom_events)
    gc.collect()
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["offline", "online"], default="offline")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--only", default=None, help="run only the configuration with this tag (e.g. to repeat a contaminated run)")
    ap.add_argument("--n-indep", type=int, default=None, help="images used for the B=1 runs (subset of the workload)")
    ap.add_argument("--big-batch-check", action="store_true", help="one B=64 run on 64 images before the repeats")
    ap.add_argument("--config", default=str(ROOT / "configs" / "local_gpu.yaml"))
    args = ap.parse_args()
    out = run_dir("exp2", args.part)
    if args.only:
        out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    cfgfile = yaml.safe_load(Path(args.config).read_text())
    backend = make_backend(cfgfile)
    rows = subset_rows(f"bench{args.n}")
    write_json(out / ("config.json" if not args.only else f"config_rerun_{args.only}.json"), {"args": vars(args), "backend": backend.info(), "environment": environment(),
                                     "uids": [r["uid"] for r in rows]})
    # warm-up (CUDA kernels, allocator); not recorded
    backend.generate_batch(make_reqs(rows[:2], "warmup"))
    if args.part == "offline":
        if args.big_batch_check:
            # single feasibility run of the largest batch on a 64-image workload (memory / OOM check)
            big = subset_rows("bench64")
            run_one(backend, big, SchedulerConfig(policy="static", batch_size=64, timeout_s=1800), out, "static_b64_check_r0")
        for rep in range(args.repeats):
            for b in args.batches:
                pol = "independent" if b == 1 else "static"
                rr = rows[: args.n_indep] if b == 1 and args.n_indep else rows
                run_one(backend, rr, SchedulerConfig(policy=pol, batch_size=b, timeout_s=1800), out, f"{pol}_b{b}_r{rep}")
            run_one(backend, rows, SchedulerConfig(policy="sorted", batch_size=16, timeout_s=1800), out, f"sorted_b16_r{rep}")
            run_one(backend, rows, SchedulerConfig(policy="adaptive", batch_size=1, max_batch_size=64,
                                                   memory_budget_mb=5800, timeout_s=1800), out, f"adaptive_r{rep}")
    else:
        for rate in args.rates:
            for rep in range(args.repeats):
                for tag, cfg_ in ((f"online_indep_{rate}Hz_r{rep}", SchedulerConfig(policy="independent", batch_size=1)),
                                  (f"online_dyn8_{rate}Hz_r{rep}", SchedulerConfig(policy="dynamic", batch_size=8, max_wait_s=2.0))):
                    if args.only and tag != args.only:
                        continue
                    run_one(backend, rows, cfg_, out, tag, rate)
    backend.close()


if __name__ == "__main__":
    main()
