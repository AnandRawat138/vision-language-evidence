"""Verify bibliography entries against Crossref (DOI) and the arXiv API.

    python scripts/verify_references.py

Reads paper/reference_candidates.json, writes results/raw/reference_verification.json with
the metadata returned by the authoritative registries (title, authors, venue, year).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "vle-reference-check/0.1 (research reproducibility script)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def crossref(doi: str) -> dict:
    d = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    return {"title": (d.get("title") or [""])[0], "container": (d.get("container-title") or [""])[0],
            "year": (d.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "authors": [f"{a.get('family', '')}, {a.get('given', '')}" for a in d.get("author", [])][:6],
            "publisher": d.get("publisher"), "type": d.get("type")}


def arxiv(aid: str) -> dict:
    x = get("http://export.arxiv.org/api/query?id_list=" + aid)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = ET.fromstring(x).find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return {"error": "not found"}
    return {"title": " ".join(e.find("a:title", ns).text.split()),
            "year": e.find("a:published", ns).text[:4],
            "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)][:6]}


def main() -> None:
    cands = json.loads((ROOT / "paper" / "reference_candidates.json").read_text(encoding="utf-8"))
    out = {}
    for key, c in cands.items():
        rec = {"claimed": c}
        try:
            if c.get("doi"):
                rec["crossref"] = crossref(c["doi"])
            if c.get("arxiv"):
                rec["arxiv"] = arxiv(c["arxiv"])
                time.sleep(3)
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)
        out[key] = rec
        t = (rec.get("crossref") or rec.get("arxiv") or {}).get("title", rec.get("error"))
        print(f"{key:28s} {t}", flush=True)
    (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "raw" / "reference_verification.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
