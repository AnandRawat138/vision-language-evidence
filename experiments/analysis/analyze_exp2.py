"""Experiment 2 analysis: throughput, latency, memory across batch policies; output invariance to batching;
online (Poisson-arrival) latency for independent vs dynamic batching.

    python experiments/analysis/analyze_exp2.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from vle.evaluation.metrics import jaccard  # noqa: E402
from vle.lexicon import map_term  # noqa: E402
from vle.parsing import parse_structured  # noqa: E402
from vle.reporting.tables import esc, fmt, write_table  # noqa: E402
from vle.runio import RAW, read_jsonl, write_json  # noqa: E402
from vle.stats import tests as S  # noqa: E402

PROC = ROOT / "results" / "processed"


def service_latencies(st: dict) -> list[float]:
    out = []
    for b, l in zip(st["batch_sizes"], st["batch_latencies"]):
        out += [l] * b
    return out


def objects_of(text: str) -> set:
    p, _, _ = parse_structured(text)
    return {map_term(o.name, "coco") for o in p.objects} - {None} if p else set()


def offline() -> dict:
    d = RAW / "exp2" / "offline"
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(d.glob("*.stats.json")):
        st = json.loads(f.read_text())
        key = re.sub(r"_r\d+$", "", st["tag"])
        groups[key].append(st)
    rows = []
    raw = {}
    for key, sts in groups.items():
        tp = np.array([s["throughput_img_s"] for s in sts])
        svc = [service_latencies(s) for s in sts]
        allsvc = np.concatenate([np.array(x) for x in svc])
        r = {"config": key, "policy": sts[0]["policy"], "batch": sts[0]["config"]["batch_size"] if sts[0]["policy"] != "adaptive" else "adaptive",
             "repeats": len(sts), "throughput_mean": float(tp.mean()), "throughput_sd": float(tp.std(ddof=1)) if len(tp) > 1 else 0.0,
             "throughput_ci": S.bootstrap_ci(tp) if len(tp) > 1 else (float(tp[0]), float(tp[0])),
             "wall_time_mean_s": float(np.mean([s["wall_time_s"] for s in sts])),
             "latency_p50": float(np.percentile(allsvc, 50)), "latency_p95": float(np.percentile(allsvc, 95)),
             "latency_p99": float(np.percentile(allsvc, 99)), "latency_mean": float(allsvc.mean()),
             "completion_p50": float(np.median(np.concatenate([s["request_latencies"] for s in sts]))),
             "gpu_mem_peak_mb": float(np.max([s.get("gpu_mem_peak_mb") or np.nan for s in sts])),
             "torch_peak_alloc_mb": float(np.max([s.get("torch_peak_alloc_mb") or np.nan for s in sts])),
             "cpu_pct_mean": float(np.mean([s.get("cpu_pct_mean") or np.nan for s in sts])),
             "ram_peak_mb": float(np.max([s.get("ram_peak_mb") or np.nan for s in sts])),
             "gpu_util_mean": float(np.mean([s.get("gpu_util_mean") or np.nan for s in sts])),
             "failed": int(sum(s["n_failed"] for s in sts)), "oom_events": int(sum(s["n_oom_events"] for s in sts)),
             "mean_batch": float(np.mean([s["mean_batch_size"] or 0 for s in sts])),
             "adaptive_final_batch": [s["adaptive_trace"][-1]["batch"] for s in sts if s.get("adaptive_trace")] or None}
        rows.append(r)
        raw[key] = tp.tolist()
    order = {"independent_b1": 0}
    rows.sort(key=lambda r: (order.get(r["config"], 1), r["policy"] == "adaptive", r["policy"] == "sorted",
                             r["batch"] if isinstance(r["batch"], int) else 999))

    # output invariance: compare each config's outputs with independent B=1 (repeat 0)
    base = {r["meta"]["uid"]: r["text"] for r in read_jsonl(d / "independent_b1_r0.jsonl")}
    inv = {}
    for f in sorted(d.glob("*_r0.jsonl")):
        tag = f.stem
        other = {r["meta"]["uid"]: r["text"] for r in read_jsonl(f)}
        common = [u for u in base if u in other]
        ident = np.mean([base[u] == other[u] for u in common])
        jac = np.mean([jaccard(objects_of(base[u]), objects_of(other[u])) for u in common])
        inv[re.sub(r"_r0$", "", tag)] = {"identical_text": float(ident), "object_set_jaccard": float(jac), "n": len(common)}
    # repeat-to-repeat determinism at the same batch size
    det = {}
    for key in groups:
        f0, f1 = d / f"{key}_r0.jsonl", d / f"{key}_r1.jsonl"
        if f0.exists() and f1.exists():
            a = {r["meta"]["uid"]: r["text"] for r in read_jsonl(f0)}
            b = {r["meta"]["uid"]: r["text"] for r in read_jsonl(f1)}
            det[key] = float(np.mean([a[u] == b[u] for u in a if u in b]))

    # H1: independent vs best static batch (Welch over repeats; small n -> also report effect size)
    best = max((r for r in rows if r["policy"] == "static"), key=lambda r: r["throughput_mean"])
    h1 = S.welch(np.array(raw[best["config"]]), np.array(raw["independent_b1"]))
    h1["speedup"] = best["throughput_mean"] / next(r for r in rows if r["config"] == "independent_b1")["throughput_mean"]
    h1["best_config"] = best["config"]
    # H4: adaptive vs static B=8 and vs best static
    h4 = {}
    if "adaptive" in raw:
        for k in ("static_b8", best["config"]):
            if k in raw:
                h4[f"adaptive vs {k}"] = {**S.welch(np.array(raw["adaptive"]), np.array(raw[k])),
                                          "ratio": float(np.mean(raw["adaptive"]) / np.mean(raw[k]))}
    return {"rows": rows, "output_invariance_vs_b1": inv, "repeat_determinism": det, "H1": h1, "H4": h4, "throughput_raw": raw}


def online() -> dict:
    d = RAW / "exp2" / "online"
    if not d.exists():
        return {"status": "not run"}
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(d.glob("*.stats.json")):
        st = json.loads(f.read_text())
        groups[re.sub(r"_r\d+$", "", st["tag"])].append(st)
    rows = []
    for k, sts in groups.items():
        lat = np.concatenate([s["request_latencies"] for s in sts])
        rows.append({"config": k, "n_runs": len(sts), "latency_p50": float(np.percentile(lat, 50)),
                     "latency_p95": float(np.percentile(lat, 95)), "latency_p99": float(np.percentile(lat, 99)),
                     "throughput_mean": float(np.mean([s["throughput_img_s"] for s in sts])),
                     "mean_batch": float(np.mean([s["mean_batch_size"] for s in sts])),
                     "lat_raw": lat.tolist()})
    tests = {}
    for r in rows:
        if r["config"].startswith("online_dyn8"):
            other = next((x for x in rows if x["config"] == r["config"].replace("dyn8", "indep")), None)
            if other:
                tests[r["config"]] = {"cliffs_delta_latency": S.cliffs_delta(np.array(r["lat_raw"]), np.array(other["lat_raw"])),
                                      "mannwhitney_p": float(__import__("scipy.stats", fromlist=["x"]).mannwhitneyu(r["lat_raw"], other["lat_raw"]).pvalue)}
    return {"rows": rows, "tests": tests}


def main() -> None:
    off = offline()
    on = online()
    summ = {"offline": off, "online": {**on, "rows": [{k: v for k, v in r.items() if k != "lat_raw"} for r in on.get("rows", [])]}}
    write_json(PROC / "exp2_full.json", {"offline": off, "online": on})
    write_json(PROC / "exp2_summary.json", {**summ, "rows": [{k: v for k, v in r.items()} for r in off["rows"]]})
    rows = []
    for r in off["rows"]:
        inv = off["output_invariance_vs_b1"].get(r["config"], {})
        rows.append([esc(r["policy"]), str(r["batch"]), f"{r['throughput_mean']:.3f} $\\pm$ {r['throughput_sd']:.3f}",
                     fmt(r["latency_p50"], 1), fmt(r["latency_p95"], 1), fmt(r["latency_p99"], 1), fmt(r["gpu_mem_peak_mb"], 0),
                     fmt(r["ram_peak_mb"], 0), fmt(r["cpu_pct_mean"], 1), str(r["failed"]), fmt(inv.get("identical_text"), 2)])
    write_table("exp2_batching", ["Policy", "$B$", "Images/s (mean $\\pm$ sd)", "p50 (s)", "p95 (s)", "p99 (s)", "GPU MB", "RAM MB",
                                  "CPU \\%", "Fail", "Ident."], rows,
                "Batch-processing scalability on the RTX~3060 Laptop GPU (6~GB): 32 images ($B{=}1$: first 16), 3 repeats per configuration; the $B{=}64$ row is a single feasibility run on 64 images. "
                "Latencies are per-request service times (batch execution time). GPU MB: peak device memory (NVML, includes the "
                "desktop). Ident.: fraction of the 16 shared images whose output text equals $B{=}1$ (larger comparison in the main text). $B{=}1$ used 16 of the 32 images.", "tab:exp2",
                colspec="lrrrrrrrrrr", size=r"\scriptsize")
    if on.get("rows"):
        rows = [[esc(r["config"].replace("online_", "")), fmt(r["throughput_mean"]), fmt(r["latency_p50"], 1), fmt(r["latency_p95"], 1),
                 fmt(r["latency_p99"], 1), fmt(r["mean_batch"], 2)] for r in on["rows"]]
        write_table("exp2_online", ["Policy @ arrival rate", "Images/s", "p50 (s)", "p95 (s)", "p99 (s)", "Mean $B$"], rows,
                    "Online processing with Poisson arrivals (24 requests, 2 repeats). Latency is end-to-end (queueing + service).",
                    "tab:exp2_online", colspec="lrrrrr", size=r"\scriptsize")
    print(json.dumps({r["config"]: (round(r["throughput_mean"], 3), round(r["latency_p95"], 1), r["gpu_mem_peak_mb"]) for r in off["rows"]}, indent=1))
    print("H1", off["H1"], "H4", off["H4"])
    print("invariance", off["output_invariance_vs_b1"])


if __name__ == "__main__":
    main()
