"""Module C: image loading, quality assessment and controlled degradation.

Source images are opened read-only; every transformation returns a new array.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from vle.schemas import ImageQuality

# Hand-chosen heuristic thresholds (NOT tuned or validated on data); indicator flags only.
BLUR_VAR_LOW = 60.0
DARK_MEAN = 50.0
LOW_CONTRAST_STD = 25.0
LOW_RES_SIDE = 320


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def assess_quality(img: Image.Image) -> ImageQuality:
    arr = np.asarray(img.convert("L"), dtype=np.uint8)
    lap = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    mean = float(arr.mean())
    std = float(arr.std())
    flags: list[str] = []
    if lap < BLUR_VAR_LOW:
        flags.append("blur")
    if mean < DARK_MEAN:
        flags.append("low_light")
    if std < LOW_CONTRAST_STD:
        flags.append("low_contrast")
    if min(img.size) < LOW_RES_SIDE:
        flags.append("low_resolution")
    return ImageQuality(width=img.size[0], height=img.size[1], blur_laplacian_var=round(lap, 2),
                        brightness_mean=round(mean, 2), contrast_std=round(std, 2), flags=flags)


# ------------------------------------------------------- degradations ------
def degrade(img: Image.Image, kind: str, level: float, seed: int = 0) -> Image.Image:
    """Controlled degradations used in Experiment 5.

    kind:  blur (Gaussian sigma in px) | downscale (scale factor, upsampled back)
           | lowlight (gamma>1 darkening + Gaussian noise) | occlusion (fraction of area, grey box)
           | jpeg (quality 1..95)
    """
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    rng = np.random.default_rng(seed)
    if kind == "blur":
        k = int(2 * round(3 * level) + 1)
        out = cv2.GaussianBlur(a, (k, k), sigmaX=level)
    elif kind == "downscale":
        nw, nh = max(8, int(w * level)), max(8, int(h * level))
        small = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "lowlight":
        f = a.astype(np.float32) / 255.0
        f = np.power(f, level) * (1.0 / level)  # gamma darkening + exposure drop
        f = f + rng.normal(0, 0.02, f.shape)
        out = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    elif kind == "occlusion":
        frac = level
        bw, bh = int(w * np.sqrt(frac)), int(h * np.sqrt(frac))
        x0 = int((w - bw) / 2)
        y0 = int((h - bh) / 2)
        out = a.copy()
        out[y0:y0 + bh, x0:x0 + bw] = 128
    elif kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=int(level))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    elif kind == "none":
        out = a
    else:
        raise ValueError(f"unknown degradation {kind}")
    return Image.fromarray(out)


def crop_view(img: Image.Image, which: str, frac: float = 0.7) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Overlapping crop used to create multi-view groups (Experiment 3). Returns (crop, box xyxy)."""
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    boxes = {"left": (0, (h - ch) // 2), "right": (w - cw, (h - ch) // 2),
             "top": ((w - cw) // 2, 0), "bottom": ((w - cw) // 2, h - ch)}
    x0, y0 = boxes[which]
    box = (x0, y0, x0 + cw, y0 + ch)
    return img.crop(box), box


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    s = max_side / max(w, h)
    if s >= 1:
        return img
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BICUBIC)
