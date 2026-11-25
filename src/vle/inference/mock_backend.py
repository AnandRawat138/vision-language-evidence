"""CPU-only deterministic backend for smoke tests and CI (no model weights).

Outputs are synthetic and are NEVER used for any reported result.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

from vle.inference.base import InferenceBackend, InferenceRequest, InferenceResult, OutOfMemory

_VOCAB = ["person", "car", "dog", "chair", "bench", "bicycle", "umbrella", "cup", "bottle", "tree"]
_CLOTH = [("shirt", "blue"), ("jacket", "black"), ("pants", "gray"), ("hat", "red")]


class MockBackend(InferenceBackend):
    name = "mock"

    def __init__(self, base_latency_s: float = 0.002, per_item_s: float = 0.001, max_batch: int = 1024,
                 fail_ids: list[str] | None = None, hang_ids: list[str] | None = None):
        self.base, self.per_item, self.max_batch = base_latency_s, per_item_s, max_batch
        self.fail_ids, self.hang_ids = set(fail_ids or []), set(hang_ids or [])

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": "mock", "model_revision": "0", "quantization": "none"}

    def _fake(self, r: InferenceRequest) -> str:
        h = int(hashlib.md5((r.text + "|".join(r.image_ids)).encode()).hexdigest(), 16)
        objs = [{"name": _VOCAB[(h >> i) % len(_VOCAB)], "confidence": ["high", "medium", "low"][(h >> (i + 3)) % 3]}
                for i in range(1 + h % 4)]
        c = _CLOTH[h % len(_CLOTH)]
        return json.dumps({"objects": objs, "clothing": [{"name": c[0], "color": c[1], "confidence": "medium"}],
                           "accessories": [], "carried_objects": [],
                           "scene": {"setting": "outdoor", "place": "street", "lighting": "daylight"},
                           "image_quality": {"blur": "none", "occlusion": "none", "lighting": "good"},
                           "unclear": []})

    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        if len(requests) > self.max_batch:
            raise OutOfMemory(f"mock OOM at {len(requests)}")
        for r in requests:
            if set(r.image_ids) & self.hang_ids:
                time.sleep(5.0)
            if set(r.image_ids) & self.fail_ids:
                raise RuntimeError(f"mock failure for {r.image_ids}")
        dt = self.base + self.per_item * len(requests)
        time.sleep(dt)
        out = []
        for r in requests:
            text = self._fake(r)
            toks = [text[i:i + 4] for i in range(0, len(text), 4)]
            out.append(InferenceResult(request_id=r.request_id, text=text, tokens=toks,
                                       token_logprobs=[math.log(0.9)] * len(toks), n_input_tokens=100,
                                       n_output_tokens=len(toks), latency_s=dt, batch_size=len(requests)))
        return out
