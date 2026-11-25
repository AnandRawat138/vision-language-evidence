"""Run directories, environment capture and output loading (all paper numbers come from these files)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"


def git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
           "git_commit": git_hash(), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        import torch
        env.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20) if torch.cuda.is_available() else None})
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil
        env["ram_total_mb"] = round(psutil.virtual_memory().total / 2**20)
        env["cpu_count_logical"] = psutil.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    return env


def run_dir(experiment: str, name: str) -> Path:
    d = RAW / experiment / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # backfill meta.uid for records written before the scheduler meta fix (uid = image id without view suffix)
    for r in rows:
        meta = r.get("meta") or {}
        if "uid" not in meta and r.get("image_ids"):
            meta["uid"] = r["image_ids"][0].split("#")[0]
        r["meta"] = meta
    # keep the last successful record per request id (resume semantics)
    best: dict[str, dict] = {}
    for r in rows:
        k = r["request_id"]
        if k not in best or (best[k].get("error") and not r.get("error")):
            best[k] = r
    return list(best.values())
