"""SQLite evidence store with an append-only audit trail of human review actions."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from vle.schemas import CorrectionEvent, Observation, ReviewStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (uid TEXT PRIMARY KEY, source TEXT, split TEXT, local_path TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS observations (
  observation_id TEXT PRIMARY KEY, image_id TEXT, run_id TEXT, category TEXT, label TEXT,
  data TEXT, review_status TEXT DEFAULT 'pending');
CREATE INDEX IF NOT EXISTS ix_obs_image ON observations(image_id);
CREATE TABLE IF NOT EXISTS audit (
  event_id TEXT PRIMARY KEY, observation_id TEXT, image_id TEXT, reviewer_id TEXT, action TEXT,
  before TEXT, after TEXT, comment TEXT, timestamp TEXT, duration_ms INTEGER);
CREATE TABLE IF NOT EXISTS image_info (uid TEXT, run_id TEXT, data TEXT, PRIMARY KEY(uid, run_id));
"""


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------------------------------------------------- ingestion --
    def upsert_images(self, rows: list[dict]) -> None:
        with self._lock:
            self.conn.executemany("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)",
                                  [(r["uid"], r["source"], r["split"], r["local_path"], json.dumps(r)) for r in rows])
            self.conn.commit()

    def add_observations(self, obs: list[Observation]) -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?)",
                [(o.observation_id, o.provenance.image_id, o.provenance.run_id, o.category.value, o.label,
                  o.model_dump_json(), o.review_status.value) for o in obs])
            self.conn.commit()

    def set_image_info(self, uid: str, run_id: str, data: dict) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO image_info VALUES (?,?,?)", (uid, run_id, json.dumps(data, default=str)))
            self.conn.commit()

    # ------------------------------------------------------------- reads ---
    def images(self, source: str | None = None, split: str | None = None, offset: int = 0, limit: int = 50) -> list[dict]:
        q, args = "SELECT uid, source, split, local_path FROM images WHERE 1=1", []
        if source:
            q += " AND source=?"
            args.append(source)
        if split:
            q += " AND split=?"
            args.append(split)
        q += " ORDER BY uid LIMIT ? OFFSET ?"
        args += [limit, offset]
        return [dict(r) for r in self.conn.execute(q, args)]

    def count_images(self) -> dict:
        rows = self.conn.execute("SELECT source, split, COUNT(*) n FROM images GROUP BY source, split").fetchall()
        return {"by_source_split": [dict(r) for r in rows], "total": sum(r["n"] for r in rows)}

    def image(self, uid: str) -> dict | None:
        r = self.conn.execute("SELECT meta FROM images WHERE uid=?", (uid,)).fetchone()
        return json.loads(r["meta"]) if r else None

    def observations(self, image_id: str, run_id: str | None = None) -> list[Observation]:
        q, args = "SELECT data, review_status FROM observations WHERE image_id=?", [image_id]
        if run_id:
            q += " AND run_id=?"
            args.append(run_id)
        out = []
        for r in self.conn.execute(q, args):
            o = Observation.model_validate_json(r["data"])
            o.review_status = ReviewStatus(r["review_status"])
            out.append(o)
        return out

    def observation(self, observation_id: str) -> Observation | None:
        r = self.conn.execute("SELECT data, review_status FROM observations WHERE observation_id=?", (observation_id,)).fetchone()
        if not r:
            return None
        o = Observation.model_validate_json(r["data"])
        o.review_status = ReviewStatus(r["review_status"])
        return o

    def image_info(self, uid: str) -> dict:
        return {r["run_id"]: json.loads(r["data"]) for r in self.conn.execute("SELECT run_id, data FROM image_info WHERE uid=?", (uid,))}

    def runs(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT run_id FROM observations ORDER BY run_id")]

    # ------------------------------------------------------------ review ---
    def review(self, ev: CorrectionEvent) -> Observation:
        """Apply a review action and append it to the audit trail (never deletes history)."""
        with self._lock:
            o = self.observation(ev.observation_id)
            if o is None and ev.action != "add":
                raise KeyError(ev.observation_id)
            if ev.action == "add":
                o = Observation.model_validate(ev.after)
                o.review_status = ReviewStatus.CORRECTED
                ev.before = None
            else:
                ev.before = o.model_dump(mode="json")
                if ev.action == "accept":
                    o.review_status = ReviewStatus.ACCEPTED
                elif ev.action == "reject":
                    o.review_status = ReviewStatus.REJECTED
                elif ev.action == "correct":
                    patch = ev.after or {}
                    for k in ("label", "attribute", "canonical"):
                        if k in patch:
                            setattr(o, k, patch[k])
                    o.review_status = ReviewStatus.CORRECTED
            ev.after = o.model_dump(mode="json")
            self.conn.execute("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?)",
                              (o.observation_id, o.provenance.image_id, o.provenance.run_id, o.category.value, o.label,
                               o.model_dump_json(), o.review_status.value))
            self.conn.execute("INSERT INTO audit VALUES (?,?,?,?,?,?,?,?,?,?)",
                              (ev.event_id, o.observation_id, o.provenance.image_id, ev.reviewer_id, ev.action,
                               json.dumps(ev.before), json.dumps(ev.after), ev.comment, ev.timestamp, ev.duration_ms))
            self.conn.commit()
            return o

    def audit(self, image_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM audit", []
        if image_id:
            q += " WHERE image_id=?"
            args.append(image_id)
        q += " ORDER BY timestamp"
        rows = []
        for r in self.conn.execute(q, args):
            d = dict(r)
            d["before"], d["after"] = json.loads(d["before"]), json.loads(d["after"])
            rows.append(d)
        return rows

    def review_stats(self) -> dict:
        rows = self.conn.execute("SELECT review_status, COUNT(*) n FROM observations GROUP BY review_status").fetchall()
        actions = self.conn.execute("SELECT action, COUNT(*) n, AVG(duration_ms) t FROM audit GROUP BY action").fetchall()
        return {"status": {r["review_status"]: r["n"] for r in rows},
                "actions": {r["action"]: {"n": r["n"], "mean_ms": r["t"]} for r in actions}}
