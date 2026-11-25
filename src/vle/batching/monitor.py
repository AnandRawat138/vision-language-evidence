"""Background resource sampler: GPU memory/utilisation (NVML), process CPU and RAM (psutil)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ResourceStats:
    samples: int = 0
    gpu_mem_mb: list[float] = field(default_factory=list)
    gpu_util: list[float] = field(default_factory=list)
    cpu_pct: list[float] = field(default_factory=list)
    ram_mb: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        def mx(x: list[float]) -> float | None:
            return round(max(x), 1) if x else None

        def mean(x: list[float]) -> float | None:
            return round(sum(x) / len(x), 1) if x else None
        return {"gpu_mem_peak_mb": mx(self.gpu_mem_mb), "gpu_util_mean": mean(self.gpu_util),
                "cpu_pct_mean": mean(self.cpu_pct), "cpu_pct_peak": mx(self.cpu_pct),
                "ram_peak_mb": mx(self.ram_mb), "samples": self.samples}


class ResourceMonitor:
    def __init__(self, interval_s: float = 0.25, gpu_index: int = 0):
        self.interval = interval_s
        self.stats = ResourceStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:  # noqa: BLE001
            self._nvml = None

    def gpu_used_mb(self) -> float | None:
        if not self._nvml:
            return None
        return self._nvml.nvmlDeviceGetMemoryInfo(self._h).used / 2**20

    def gpu_total_mb(self) -> float | None:
        if not self._nvml:
            return None
        return self._nvml.nvmlDeviceGetMemoryInfo(self._h).total / 2**20

    def _run(self) -> None:
        self._proc.cpu_percent(None)
        while not self._stop.is_set():
            s = self.stats
            s.samples += 1
            s.cpu_pct.append(self._proc.cpu_percent(None) / psutil.cpu_count())
            s.ram_mb.append(self._proc.memory_info().rss / 2**20)
            if self._nvml:
                s.gpu_mem_mb.append(self._nvml.nvmlDeviceGetMemoryInfo(self._h).used / 2**20)
                s.gpu_util.append(float(self._nvml.nvmlDeviceGetUtilizationRates(self._h).gpu))
            time.sleep(self.interval)

    def __enter__(self) -> "ResourceMonitor":
        self.stats = ResourceStats()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
