"""vLLM backend through the OpenAI-compatible server (``vllm serve``).

Intended for Linux/WSL2 or cloud GPUs. Start a server, e.g.

    vllm serve Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 8192 \
        --limit-mm-per-prompt '{"image":4}' --mm-processor-kwargs '{"max_pixels":200704}'

then set ``backend: vllm`` in the experiment config.  vLLM performs continuous
batching on the server; this client submits ``len(requests)`` concurrent
requests so the scheduler-level batch size maps to request concurrency.
Not executed on the local Windows laptop (see docs/feasibility_report.md).
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from vle.inference.base import InferenceBackend, InferenceRequest, InferenceResult


class VLLMOpenAIBackend(InferenceBackend):
    name = "vllm-openai"

    def __init__(self, base_url: str = "http://localhost:8000/v1", model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                 api_key: str = "EMPTY", timeout_s: float = 120.0, repetition_penalty: float = 1.05,
                 quantization: str = "none"):
        self.base_url, self.model_id, self.api_key = base_url.rstrip("/"), model_id, api_key
        self.timeout_s, self.repetition_penalty, self.quantization = timeout_s, repetition_penalty, quantization

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "base_url": self.base_url,
                "quantization": self.quantization, "decoding": "greedy",
                "repetition_penalty": self.repetition_penalty}

    @staticmethod
    def _b64(img) -> str:  # type: ignore[no-untyped-def]
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _one(self, r: InferenceRequest) -> InferenceResult:
        content: list[dict] = [{"type": "image_url", "image_url": {"url": self._b64(im)}} for im in r.images]
        content.append({"type": "text", "text": r.text})
        msgs = ([{"role": "system", "content": r.system}] if r.system else []) + [{"role": "user", "content": content}]
        body = {"model": self.model_id, "messages": msgs, "max_tokens": r.max_new_tokens, "temperature": 0.0,
                "repetition_penalty": self.repetition_penalty, "logprobs": True, "top_logprobs": 0}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read())
        except Exception as e:  # noqa: BLE001
            return InferenceResult(request_id=r.request_id, text="", error=str(e)[:300], latency_s=time.perf_counter() - t0)
        ch = data["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        return InferenceResult(request_id=r.request_id, text=ch["message"]["content"] or "",
                               tokens=[t["token"] for t in lp], token_logprobs=[t["logprob"] for t in lp],
                               n_input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                               n_output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                               latency_s=time.perf_counter() - t0)

    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        with ThreadPoolExecutor(max_workers=max(1, len(requests))) as ex:
            res = list(ex.map(self._one, requests))
        for r in res:
            r.batch_size = len(requests)
        return res
