"""Local Hugging Face Transformers backend (used on the 6 GB laptop GPU).

vLLM does not run natively on Windows and was not used on the local machine;
this backend provides batched, left-padded greedy generation with per-token
log-probabilities so that results are comparable with the vLLM backend, which
returns the same fields through the OpenAI-compatible API.
"""

from __future__ import annotations

import logging
import platform
import time
from typing import Any

import torch

from vle.inference.base import InferenceBackend, InferenceRequest, InferenceResult, OutOfMemory

log = logging.getLogger(__name__)


class _LogprobRecorder:
    """Logits processor (placed last) that records log p(chosen token) per step without storing
    full-vocabulary score tensors (which would need ~B*V*T*4 bytes of GPU memory).

    At step t it receives the processed logits for position t; the token chosen from them is only
    visible at step t+1 (as input_ids[:, -1]), so one step of log-softmax is kept."""

    def __init__(self) -> None:
        self.prev: torch.Tensor | None = None
        self.out: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prev is not None:
            self.out.append(self.prev.gather(1, input_ids[:, -1:]).squeeze(1).float().cpu())
        self.prev = torch.log_softmax(scores.float(), dim=-1)
        return scores

    def finalize(self, sequences: torch.LongTensor) -> torch.Tensor:
        if self.prev is not None:
            self.out.append(self.prev.gather(1, sequences[:, -1:]).squeeze(1).float().cpu())
            self.prev = None
        return torch.stack(self.out, dim=1) if self.out else torch.zeros((sequences.shape[0], 0))


class HFVisionLanguageBackend(InferenceBackend):
    name = "hf-transformers"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", revision: str | None = None,
                 quantization: str = "nf4", dtype: str = "float16", min_pixels: int = 128 * 28 * 28,
                 max_pixels: int = 448 * 448, repetition_penalty: float = 1.05,
                 attn_implementation: str = "sdpa", device: str = "cuda", compute_logprobs: bool = True):
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        self.model_id, self.revision, self.quantization = model_id, revision, quantization
        self.min_pixels, self.max_pixels = min_pixels, max_pixels
        self.repetition_penalty = repetition_penalty
        self.compute_logprobs = compute_logprobs
        self.device = device
        torch_dtype = getattr(torch, dtype)
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "attn_implementation": attn_implementation}
        if quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": 0}
        elif quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["device_map"] = {"": 0} if device == "cuda" else None
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, min_pixels=min_pixels,
                                                       max_pixels=max_pixels, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, **kwargs).eval()
        self.load_time_s = time.perf_counter() - t0
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or revision
        log.info("loaded %s (%s) in %.1fs", model_id, quantization, self.load_time_s)

    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        import transformers
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        try:
            import bitsandbytes as bnb
            bnb_v = bnb.__version__
        except Exception:  # noqa: BLE001
            bnb_v = None
        return {"backend": self.name, "model_id": self.model_id, "model_revision": self.resolved_revision,
                "quantization": self.quantization, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels,
                "decoding": "greedy", "repetition_penalty": self.repetition_penalty,
                "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                "bitsandbytes": bnb_v, "gpu": gpu, "os": platform.platform(), "load_time_s": round(self.load_time_s, 2)}

    def memory_used_mb(self) -> float | None:
        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / 2**20

    def max_images_per_request(self) -> int:
        return 4

    # ------------------------------------------------------------------
    def _messages(self, r: InferenceRequest) -> list[dict]:
        content: list[dict] = [{"type": "image"} for _ in r.images]
        content.append({"type": "text", "text": r.text})
        msgs = []
        if r.system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": r.system}]})
        msgs.append({"role": "user", "content": content})
        return msgs

    @torch.inference_mode()
    def generate_batch(self, requests: list[InferenceRequest]) -> list[InferenceResult]:
        texts = [self.processor.apply_chat_template(self._messages(r), add_generation_prompt=True, tokenize=False)
                 for r in requests]
        images = [im for r in requests for im in r.images]
        max_new = max(r.max_new_tokens for r in requests)
        t0 = time.perf_counter()
        from transformers import LogitsProcessorList
        rec = _LogprobRecorder() if self.compute_logprobs else None
        try:
            inputs = self.processor(text=texts, images=images or None, padding=True, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, repetition_penalty=self.repetition_penalty,
                                      return_dict_in_generate=True,
                                      logits_processor=LogitsProcessorList([rec]) if rec else None)
        except torch.cuda.OutOfMemoryError as e:
            del e
            rec = None
            torch.cuda.empty_cache()
            raise OutOfMemory(f"OOM at batch={len(requests)}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        in_len = inputs["input_ids"].shape[1]
        gen = out.sequences[:, in_len:]
        if rec is not None:
            trans = rec.finalize(out.sequences)
        tok = self.processor.tokenizer
        eos_ids = set([tok.eos_token_id, tok.pad_token_id] + list(getattr(self.model.generation_config, "eos_token_id", []) or []))
        results = []
        attn = inputs["attention_mask"]
        for i, r in enumerate(requests):
            ids = gen[i].tolist()
            n = len(ids)
            while n > 0 and ids[n - 1] in eos_ids:
                n -= 1
            ids = ids[:n]
            pieces = [tok.decode([t], skip_special_tokens=True) for t in ids]
            lps = trans[i, :n].tolist() if self.compute_logprobs else []
            results.append(InferenceResult(
                request_id=r.request_id, text="".join(pieces), tokens=pieces, token_logprobs=lps,
                n_input_tokens=int(attn[i].sum()), n_output_tokens=n, latency_s=dt, batch_size=len(requests)))
        del out, inputs
        return results

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
