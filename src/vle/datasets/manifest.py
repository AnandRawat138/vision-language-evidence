"""Module A: dataset manager — sampling, download, validation, dedup, splits.

Usage (see scripts/prepare_data.py):
    build_selection(...)  -> candidate list per source (oversampled)
    download_images(...)  -> local copies (originals are never modified)
    finalize_manifest(...) -> dedup + validation + split assignment + manifest CSV/JSON
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from vle.lexicon import map_term

log = logging.getLogger(__name__)

SEED = 2026
ALLOCATION = {"coco": 500, "vg": 250, "flickr30k": 150, "vqa": 100}
OVERSAMPLE = 1.15
DEV_FRACTION = 0.30
FLICKR_ZIP_URL = "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"


@dataclass
class ManifestEntry:
    uid: str                      # framework image identifier, e.g. coco_000000000139
    source: str                   # coco | vg | flickr30k | vqa
    source_id: str                # identifier in the original dataset
    source_split: str             # original split (val2017, train2017, VG1.2, test, ...)
    url: str
    local_path: str = ""
    md5: str = ""
    dhash: str = ""
    width: int = 0
    height: int = 0
    license: str = ""
    split: str = ""               # dev | test (this project)
    has_person: bool | None = None
    status: str = "candidate"     # candidate | ok | duplicate | invalid | download_failed | surplus
    exclusion_reason: str = ""
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------- selection ---
def select_coco(coco: dict, n: int, rng: random.Random) -> list[ManifestEntry]:
    ids = sorted(i for i, anns in coco["instances"].items() if anns and i in coco["captions"])
    person = [i for i in ids if any(a["category"] == "person" for a in coco["instances"][i])]
    nonperson = [i for i in ids if i not in set(person)]
    k_p = int(round(n * OVERSAMPLE * 0.6))
    k_n = int(round(n * OVERSAMPLE * 0.4))
    chosen = rng.sample(person, k_p) + rng.sample(nonperson, k_n)
    out = []
    for i in chosen:
        im = coco["images"][i]
        lic = coco["licenses"].get(im.get("license"), {}).get("name", "")
        out.append(ManifestEntry(uid=f"coco_{i:012d}", source="coco", source_id=str(i), source_split="val2017",
                                 url=im["coco_url"], license=lic, has_person=i in set(person)))
    return out


def select_vqa(vqa: dict[int, list], coco_train: dict, exclude_ids: set[int], n: int,
               rng: random.Random) -> list[ManifestEntry]:
    ids = sorted(i for i, qs in vqa.items() if len(qs) >= 3 and i in coco_train["images"]
                 and i not in exclude_ids and coco_train["instances"].get(i))
    chosen = rng.sample(ids, int(round(n * OVERSAMPLE)))
    out = []
    for i in chosen:
        im = coco_train["images"][i]
        lic = coco_train["licenses"].get(im.get("license"), {}).get("name", "")
        has_p = any(a["category"] == "person" for a in coco_train["instances"][i])
        out.append(ManifestEntry(uid=f"vqa_{i:012d}", source="vqa", source_id=str(i),
                                 source_split="val2014 (COCO train2017 file)", url=im["coco_url"],
                                 license=lic, has_person=has_p))
    return out


def select_vg(image_data: list[dict], vg_attr_index: dict[int, list], n: int,
              rng: random.Random) -> list[ManifestEntry]:
    """Non-COCO VG images that contain >=1 annotated clothing object."""
    cands = []
    for im in image_data:
        if im.get("coco_id"):
            continue
        objs = vg_attr_index.get(im["image_id"])
        if not objs:
            continue
        if any(map_term(nm, "clothing") for o in objs for nm in o.get("names", [])):
            cands.append(im)
    cands.sort(key=lambda d: d["image_id"])
    chosen = rng.sample(cands, int(round(n * OVERSAMPLE)))
    return [ManifestEntry(uid=f"vg_{im['image_id']}", source="vg", source_id=str(im["image_id"]),
                          source_split="VG1.2", url=im["url"], license="CC BY 4.0 (Visual Genome)",
                          has_person=None) for im in chosen]


def select_flickr(test_ids: list[str], n: int, rng: random.Random) -> list[ManifestEntry]:
    ids = sorted(test_ids)
    chosen = rng.sample(ids, int(round(n * OVERSAMPLE)))
    return [ManifestEntry(uid=f"flickr30k_{i}", source="flickr30k", source_id=i, source_split="test (Entities)",
                          url=f"{FLICKR_ZIP_URL}#flickr30k-images/{i}.jpg",
                          license="Flickr terms of use; research use (Flickr30K)", has_person=None)
            for i in chosen]


# -------------------------------------------------------------- download ---
def _fetch(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    import urllib.request
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vle-research-dataset-prep/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def download_images(entries: list[ManifestEntry], image_root: Path, workers: int = 8) -> None:
    from concurrent.futures import ThreadPoolExecutor

    image_root.mkdir(parents=True, exist_ok=True)
    flickr = [e for e in entries if e.source == "flickr30k"]
    web = [e for e in entries if e.source != "flickr30k"]

    def one(e: ManifestEntry) -> None:
        dst = image_root / e.source / f"{e.uid}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                dst.write_bytes(_fetch(e.url))
            except Exception as ex:  # noqa: BLE001
                e.status, e.exclusion_reason = "download_failed", str(ex)[:200]
                log.warning("download failed %s", e.uid)
                return
        e.local_path = str(dst.relative_to(image_root.parent.parent)).replace("\\", "/")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, web))

    if flickr:
        # Range-request extraction of individual members from the 4.4 GB archive.
        from remotezip import RemoteZip
        todo = [e for e in flickr if not (image_root / "flickr30k" / f"{e.uid}.jpg").exists()]
        if todo:
            with RemoteZip(FLICKR_ZIP_URL) as rz:
                names = {Path(n).name: n for n in rz.namelist()}
                for e in todo:
                    member = names.get(f"{e.source_id}.jpg")
                    dst = image_root / "flickr30k" / f"{e.uid}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if member is None:
                        e.status, e.exclusion_reason = "download_failed", "not in archive"
                        continue
                    try:
                        dst.write_bytes(rz.read(member))
                    except Exception as ex:  # noqa: BLE001
                        e.status, e.exclusion_reason = "download_failed", str(ex)[:200]
        for e in flickr:
            dst = image_root / "flickr30k" / f"{e.uid}.jpg"
            if dst.exists():
                e.local_path = str(dst.relative_to(image_root.parent.parent)).replace("\\", "/")


# ------------------------------------------------------ validation/dedup ---
def dhash(img: Image.Image, size: int = 8) -> str:
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def validate_and_hash(e: ManifestEntry, project_root: Path, min_side: int = 64) -> None:
    if e.status == "download_failed":
        return
    p = project_root / e.local_path
    try:
        raw = p.read_bytes()
        img = Image.open(io.BytesIO(raw))
        img.verify()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as ex:  # noqa: BLE001
        e.status, e.exclusion_reason = "invalid", f"unreadable: {ex}"[:200]
        return
    e.md5 = hashlib.md5(raw).hexdigest()
    e.width, e.height = img.size
    if min(img.size) < min_side:
        e.status, e.exclusion_reason = "invalid", f"too small {img.size}"
        return
    e.dhash = dhash(img)
    e.status = "ok"


def deduplicate(entries: list[ManifestEntry], max_hamming: int = 4) -> None:
    """Exact (md5) and near-duplicate (dHash) removal; first occurrence in priority order wins."""
    kept: list[ManifestEntry] = []
    seen_md5: dict[str, str] = {}
    for e in entries:
        if e.status != "ok":
            continue
        if e.md5 in seen_md5:
            e.status, e.exclusion_reason = "duplicate", f"md5 == {seen_md5[e.md5]}"
            continue
        dup = next((k for k in kept if hamming(k.dhash, e.dhash) <= max_hamming), None)
        if dup is not None:
            e.status, e.exclusion_reason = "duplicate", f"dhash~{dup.uid} (d={hamming(dup.dhash, e.dhash)})"
            continue
        seen_md5[e.md5] = e.uid
        kept.append(e)


def trim_and_split(entries: list[ManifestEntry], rng: random.Random) -> None:
    """Keep exactly ALLOCATION[source] valid images (in sampled order) and assign dev/test."""
    for src, n in ALLOCATION.items():
        ok = [e for e in entries if e.source == src and e.status == "ok"]
        for e in ok[n:]:
            e.status, e.exclusion_reason = "surplus", "oversampled reserve not needed"
        final = ok[:n]
        if len(final) < n:
            log.warning("source %s has only %d valid images (target %d)", src, len(final), n)
        order = list(range(len(final)))
        rng.shuffle(order)
        n_dev = int(round(len(final) * DEV_FRACTION))
        for rank, idx in enumerate(order):
            final[idx].split = "dev" if rank < n_dev else "test"


def write_manifest(entries: list[ManifestEntry], out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(e) for e in entries]
    with open(out_dir / "manifest_all_candidates.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "entries": rows}, f, indent=1)
    final = [r for r in rows if r["status"] == "ok"]
    cols = ["uid", "source", "source_id", "source_split", "split", "url", "local_path", "md5", "dhash",
            "width", "height", "has_person", "license"]
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(final)
    excl = [r for r in rows if r["status"] not in ("ok",)]
    with open(out_dir / "exclusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "source", "source_id", "status", "exclusion_reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(excl)


def load_manifest(project_root: Path, split: str | None = None, source: str | None = None) -> list[dict]:
    with open(project_root / "data" / "manifests" / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    if source:
        rows = [r for r in rows if r["source"] == source]
    return rows
