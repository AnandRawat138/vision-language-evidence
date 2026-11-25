"""Load the manifest and the scored Exp-1 observations into the SQLite review store used by the dashboard.

    python scripts/ingest_review_store.py
    uvicorn vle.api.app:app --port 8008 --app-dir src      # then open http://127.0.0.1:8008
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vle.api.store import EvidenceStore  # noqa: E402
from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.schemas import Observation  # noqa: E402


def main() -> None:
    db = ROOT / "results" / "evidence.sqlite"
    s = EvidenceStore(db)
    s.upsert_images(load_manifest(ROOT))
    p = ROOT / "results" / "processed" / "exp1_observations.jsonl"
    obs = [Observation.model_validate_json(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    s.add_observations(obs)
    info = json.loads((ROOT / "results" / "processed" / "exp1_image_info.json").read_text())
    run_id = obs[0].provenance.run_id if obs else "exp1"
    for uid, d in info.items():
        s.set_image_info(uid, run_id, d)
    print(f"ingested {len(obs)} observations for {len(info)} images into {db}")


if __name__ == "__main__":
    main()
