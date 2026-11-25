"""Module E: batch scheduler for *independent* VLM requests.

Policies
--------
independent  one request per backend call, sequential (baseline)
static       fixed batch size B, requests taken in submission order
sorted       fixed B after length-aware bucketing (sort by visual-token estimate)
dynamic      online: requests arrive over time (e.g. Poisson); a batch is
             dispatched when B_max requests are queued or the oldest request has
             waited ``max_wait_s``; reports queueing + service latency
adaptive     multiplicative-increase/multiplicative-decrease controller: doubles B while throughput improves and the
             memory/latency budgets hold, halves B on OOM or budget violation

Robustness
----------
* OOM -> batch split in half and retried (down to 1); the cap is lowered.
* Exceptions -> bisection to isolate the failing request; others are retried.
* Per-request retry budget; permanent failures are recorded, never dropped silently.
* Watchdog timeout per backend call (results marked ``timeout``).
* JSONL checkpoint: completed request ids are skipped on resume.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from vle.batching.monitor import ResourceMonitor
from vle.inference.base import InferenceBackend, InferenceRequest, InferenceResult, OutOfMemory

log = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    policy: str = "static"            # independent | static | sorted | dynamic | adaptive
    batch_size: int = 8               # B (static/sorted) or B_max (dynamic) or B_init (adaptive)
    max_batch_size: int = 64          # adaptive upper bound
    max_wait_s: float = 0.5           # dynamic
    timeout_s: float = 600.0          # per backend call
    max_retries: int = 2
    memory_budget_mb: float | None = None   # adaptive: keep device memory under this
    latency_slo_s: float | None = None      # adaptive: keep per-call latency under this
    growth_tolerance: float = 0.03          # adaptive: min relative throughput gain to keep growing
    checkpoint_path: str | None = None


@dataclass
class RunStats:
    policy: str
    n_requests: int = 0
    n_ok: int = 0
    n_failed: int = 0
    n_timeout: int = 0
    n_oom_events: int = 0
    n_retries: int = 0
    wall_time_s: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)
    batch_latencies: list[float] = field(default_factory=list)
    request_latencies: list[float] = field(default_factory=list)   # queue wait + service
    resources: dict = field(default_factory=dict)
    adaptive_trace: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        import numpy as np
        lat = np.array(self.request_latencies) if self.request_latencies else np.array([float("nan")])
        return {
            "policy": self.policy, "n_requests": self.n_requests, "n_ok": self.n_ok, "n_failed": self.n_failed,
            "n_timeout": self.n_timeout, "n_oom_events": self.n_oom_events, "n_retries": self.n_retries,
            "wall_time_s": round(self.wall_time_s, 3),
            "throughput_img_s": round(self.n_ok / self.wall_time_s, 4) if self.wall_time_s else None,
            "latency_median_s": float(np.nanmedian(lat)), "latency_p95_s": float(np.nanpercentile(lat, 95)),
            "latency_p99_s": float(np.nanpercentile(lat, 99)), "latency_mean_s": float(np.nanmean(lat)),
            "mean_batch_size": float(np.mean(self.batch_sizes)) if self.batch_sizes else None,
            **self.resources,
        }


def visual_cost(r: InferenceRequest) -> int:
    return sum(im.size[0] * im.size[1] for im in r.images) + 4 * len(r.text)


class BatchScheduler:
    def __init__(self, backend: InferenceBackend, cfg: SchedulerConfig,
                 on_result: Callable[[InferenceRequest, InferenceResult], None] | None = None):
        self.backend, self.cfg, self.on_result = backend, cfg, on_result
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._done_ids: set[str] = set()
        self._ckpt = None
        if cfg.checkpoint_path:
            p = Path(cfg.checkpoint_path)
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        d = json.loads(line)
                        if d.get("error") is None:
                            self._done_ids.add(d["request_id"])
                    except json.JSONDecodeError:
                        continue
            p.parent.mkdir(parents=True, exist_ok=True)
            self._ckpt = open(p, "a", encoding="utf-8")

    # ---------------------------------------------------------- core call --
    def _call(self, batch: list[InferenceRequest], stats: RunStats) -> list[InferenceResult]:
        fut = self._pool.submit(self.backend.generate_batch, batch)
        try:
            return fut.result(timeout=self.cfg.timeout_s)
        except FutTimeout:
            stats.n_timeout += len(batch)
            # The worker thread cannot be killed; replace the pool so later calls proceed.
            self._pool = ThreadPoolExecutor(max_workers=1)
            return [InferenceResult(request_id=r.request_id, text="", error="timeout", batch_size=len(batch))
                    for r in batch]

    def _run_batch(self, batch: list[InferenceRequest], stats: RunStats, attempt: int = 0) -> list[InferenceResult]:
        """Execute with OOM splitting and failure bisection. Always returns len(batch) results."""
        t0 = time.perf_counter()
        try:
            res = self._call(batch, stats)
        except OutOfMemory:
            stats.n_oom_events += 1
            if len(batch) == 1:
                return [InferenceResult(request_id=batch[0].request_id, text="", error="oom", batch_size=1)]
            mid = len(batch) // 2
            self.cfg.batch_size = max(1, min(self.cfg.batch_size, mid))
            return self._run_batch(batch[:mid], stats, attempt) + self._run_batch(batch[mid:], stats, attempt)
        except Exception as e:  # noqa: BLE001
            log.warning("batch of %d failed: %s", len(batch), e)
            if len(batch) == 1:
                if attempt < self.cfg.max_retries:
                    stats.n_retries += 1
                    return self._run_batch(batch, stats, attempt + 1)
                return [InferenceResult(request_id=batch[0].request_id, text="", error=f"error: {e}"[:300])]
            mid = len(batch) // 2
            return self._run_batch(batch[:mid], stats, attempt) + self._run_batch(batch[mid:], stats, attempt)
        dt = time.perf_counter() - t0
        stats.batch_sizes.append(len(batch))
        stats.batch_latencies.append(dt)
        return res

    def _emit(self, batch: list[InferenceRequest], res: list[InferenceResult], stats: RunStats,
              submit_times: dict[str, float]) -> None:
        now = time.perf_counter()
        for r, x in zip(batch, res):
            st = submit_times.get(r.request_id)
            if st is not None:
                total = now - st
                x.queue_wait_s = max(0.0, total - x.latency_s)
                stats.request_latencies.append(total)
            if x.ok:
                stats.n_ok += 1
            else:
                stats.n_failed += 1
            if self._ckpt:
                rec = {"request_id": r.request_id, "image_ids": r.image_ids, "prompt_id": r.prompt_id, **_result_dict(x)}
                rec["meta"] = {**(x.meta or {}), **(r.meta or {})}   # request metadata must not be overwritten
                self._ckpt.write(json.dumps(rec) + "\n")
                self._ckpt.flush()
            if self.on_result:
                self.on_result(r, x)

    # ---------------------------------------------------------- offline ----
    def run(self, requests: list[InferenceRequest]) -> RunStats:
        """Offline policies: all requests available at t=0 (latency = completion time since start)."""
        reqs = [r for r in requests if r.request_id not in self._done_ids]
        stats = RunStats(policy=self.cfg.policy, n_requests=len(reqs))
        pol = self.cfg.policy
        if pol == "dynamic":
            raise ValueError("use run_online() for the dynamic policy")
        if pol == "sorted":
            reqs = sorted(reqs, key=visual_cost)
        t_start = time.perf_counter()
        submit = {r.request_id: t_start for r in reqs}
        with ResourceMonitor() as mon:
            if pol == "adaptive":
                self._run_adaptive(reqs, stats, submit)
            else:
                b = 1 if pol == "independent" else self.cfg.batch_size
                i = 0
                while i < len(reqs):
                    b = 1 if pol == "independent" else self.cfg.batch_size
                    batch = reqs[i:i + b]
                    res = self._run_batch(batch, stats)
                    self._emit(batch, res, stats, submit)
                    i += len(batch)
        stats.wall_time_s = time.perf_counter() - t_start
        stats.resources = mon.stats.summary()
        torch_peak = self.backend.memory_used_mb()
        if torch_peak is not None:
            stats.resources["torch_peak_alloc_mb"] = round(torch_peak, 1)
        return stats

    def _run_adaptive(self, reqs: list[InferenceRequest], stats: RunStats, submit: dict[str, float]) -> None:
        cfg = self.cfg
        b = max(1, cfg.batch_size)
        best_tp = 0.0
        growing = True
        i = 0
        reqs = sorted(reqs, key=visual_cost)  # length-aware bucketing
        mon_gpu = ResourceMonitor()
        while i < len(reqs):
            batch = reqs[i:i + b]
            t0 = time.perf_counter()
            n_oom_before = stats.n_oom_events
            res = self._run_batch(batch, stats)
            dt = time.perf_counter() - t0
            self._emit(batch, res, stats, submit)
            i += len(batch)
            tp = sum(1 for x in res if x.ok) / dt if dt > 0 else 0.0
            mem = mon_gpu.gpu_used_mb()
            violated = (stats.n_oom_events > n_oom_before
                        or (cfg.memory_budget_mb is not None and mem is not None and mem > cfg.memory_budget_mb)
                        or (cfg.latency_slo_s is not None and dt > cfg.latency_slo_s))
            stats.adaptive_trace.append({"batch": len(batch), "latency_s": round(dt, 3), "throughput": round(tp, 3),
                                         "gpu_mem_mb": round(mem, 1) if mem else None, "violated": bool(violated)})
            if violated:
                b = max(1, len(batch) // 2)
                growing = False
            elif growing:
                if tp > best_tp * (1 + cfg.growth_tolerance):
                    best_tp = tp
                    b = min(cfg.max_batch_size, b * 2)
                else:
                    growing = False
                    b = max(1, b // 2) if tp < best_tp * 0.97 else b
            self.cfg.batch_size = b

    # ----------------------------------------------------------- online ----
    def run_online(self, requests: list[InferenceRequest], arrival_rate_hz: float, seed: int = 0) -> RunStats:
        """Dynamic batching with Poisson arrivals at ``arrival_rate_hz`` (real wall-clock)."""
        stats = RunStats(policy=f"{self.cfg.policy}@{arrival_rate_hz}Hz", n_requests=len(requests))
        rng = random.Random(seed)
        q: queue.Queue = queue.Queue()
        submit: dict[str, float] = {}
        t_start = time.perf_counter()

        def producer() -> None:
            t = t_start
            for r in requests:
                t += rng.expovariate(arrival_rate_hz)
                delay = t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                submit[r.request_id] = time.perf_counter()
                q.put(r)
            q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        finished = False
        with ResourceMonitor() as mon:
            th.start()
            bmax = 1 if self.cfg.policy == "independent" else self.cfg.batch_size
            while not finished:
                batch: list[InferenceRequest] = []
                item = q.get()
                if item is None:
                    break
                batch.append(item)
                deadline = submit[item.request_id] + (0 if bmax == 1 else self.cfg.max_wait_s)
                while len(batch) < bmax:
                    timeout = deadline - time.perf_counter()
                    if timeout <= 0:
                        # drain anything already waiting without blocking
                        try:
                            nxt = q.get_nowait()
                        except queue.Empty:
                            break
                    else:
                        try:
                            nxt = q.get(timeout=timeout)
                        except queue.Empty:
                            break
                    if nxt is None:
                        finished = True
                        break
                    batch.append(nxt)
                res = self._run_batch(batch, stats)
                self._emit(batch, res, stats, submit)
        stats.wall_time_s = time.perf_counter() - t_start
        stats.resources = mon.stats.summary()
        return stats

    def close(self) -> None:
        if self._ckpt:
            self._ckpt.close()
        self._pool.shutdown(wait=False)


def _result_dict(x: InferenceResult) -> dict:
    d = asdict(x)
    d["token_logprobs"] = [round(v, 4) for v in d["token_logprobs"]]
    return d


def chunked(xs: Iterable, n: int):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


__all__ = ["BatchScheduler", "SchedulerConfig", "RunStats", "chunked", "math"]
