"""Module D: common inference interface shared by local, vLLM and cloud backends.

A backend receives a *list of independent requests* (``generate_batch``).  Each
request may itself contain several images; a multi-image request is a single
contextual reasoning call, which is a different thing from batching several
independent requests together.  The two are kept separate throughout the code:

* batch size   = number of independent requests executed together
* images/request = number of images in one contextual prompt
"""

from __future__ import annotations

import abc
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class InferenceRequest:
    prompt_id: str
    text: str
    images: list[Image.Image] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    system: str | None = None
    max_new_tokens: int = 384
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    request_id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    latency_s: float = 0.0          # wall time of the batch call this request was part of
    queue_wait_s: float = 0.0       # time spent waiting in the scheduler queue
    batch_size: int = 1
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def span_confidence(self, start_char: int, end_char: int) -> float | None:
        """Geometric-mean token probability of tokens overlapping [start_char, end_char)."""
        if not self.tokens or not self.token_logprobs:
            return None
        pos = 0
        lps = []
        for tok, lp in zip(self.tokens, self.token_logprobs):
            nxt = pos + len(tok)
            if nxt > start_char and pos < end_char and tok.strip():
                lps.append(lp)
            pos = nxt
            if pos >= end_char:
                break
        if not lps:
            return None
        return float(math.exp(sum(lps) / len(lps)))


class InferenceBackend(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        """Run independent requests together. Must return results in input order."""

    @abc.abstractmethod
    def info(self) -> dict[str, Any]:
        """Model name, revision, quantization, runtime versions, generation settings."""

    def max_images_per_request(self) -> int:
        return 4

    def memory_used_mb(self) -> float | None:
        return None

    def close(self) -> None:  # pragma: no cover - optional
        pass


class OutOfMemory(RuntimeError):
    """Raised by a backend when a batch does not fit in accelerator memory."""
