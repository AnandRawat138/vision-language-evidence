"""CLIP image-text grounding scores (independent visual verifier of VLM claims).

Also provides the CLIP zero-shot multi-label baseline (B2)."""

from __future__ import annotations

import torch
from PIL import Image

PROMPT_TEMPLATE = "a photo of a {}."


class ClipScorer:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self._text_cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def image_emb(self, images: list[Image.Image]) -> torch.Tensor:
        px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.model.dtype)
        e = self.model.get_image_features(pixel_values=px)
        return torch.nn.functional.normalize(e.float(), dim=-1)

    @torch.inference_mode()
    def text_emb(self, labels: list[str]) -> torch.Tensor:
        missing = [l for l in labels if l not in self._text_cache]
        if missing:
            tok = self.proc(text=[PROMPT_TEMPLATE.format(l) for l in missing], return_tensors="pt", padding=True).to(self.device)
            e = torch.nn.functional.normalize(self.model.get_text_features(**tok).float(), dim=-1)
            for l, v in zip(missing, e):
                self._text_cache[l] = v
        return torch.stack([self._text_cache[l] for l in labels])

    def scores(self, image: Image.Image, labels: list[str]) -> list[float]:
        if not labels:
            return []
        return (self.image_emb([image]) @ self.text_emb(labels).T)[0].tolist()
