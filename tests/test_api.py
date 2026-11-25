"""API endpoint tests against a temporary SQLite store."""

from __future__ import annotations

import importlib
import time

from fastapi.testclient import TestClient

from vle.schemas import Category, Observation, Provenance, UncertaintyInfo


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("VLE_DB", str(tmp_path / "t.sqlite"))
    import vle.api.app as appmod
    appmod = importlib.reload(appmod)
    s = appmod.store
    s.upsert_images([{"uid": "coco_000000000139", "source": "coco", "source_id": "139", "split": "dev",
                      "local_path": "data/images/coco/coco_000000000139.jpg", "md5": "x"}])
    s.add_observations([Observation(category=Category.CLOTHING, label="shirt", canonical="shirt", raw_text="t-shirt",
                                    attribute="red", observation_id="o1",
                                    provenance=Provenance(source_type="vlm", source_id="coco_000000000139",
                                                          image_id="coco_000000000139", run_id="r1", model_name="m"),
                                    uncertainty=UncertaintyInfo(token_confidence=0.8))])
    return TestClient(appmod.app), appmod


def test_overview_and_image(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    assert c.get("/api/health").json()["ok"]
    ov = c.get("/api/overview").json()
    assert ov["images"]["total"] == 1 and ov["runs"] == ["r1"]
    d = c.get("/api/images/coco_000000000139").json()
    assert d["observations"][0]["label"] == "shirt"
    assert c.get("/api/images/nope").status_code == 404


def test_review_audit_trail(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    r = c.post("/api/observations/o1/review", json={"reviewer_id": "rev1", "action": "correct",
                                                    "after": {"attribute": "blue"}, "comment": "colour", "duration_ms": 1200})
    assert r.status_code == 200 and r.json()["observation"]["attribute"] == "blue"
    c.post("/api/observations/o1/review", json={"reviewer_id": "rev2", "action": "accept"})
    a = c.get("/api/audit", params={"image_id": "coco_000000000139"}).json()
    assert [x["action"] for x in a] == ["correct", "accept"]
    assert a[0]["before"]["attribute"] == "red" and a[0]["after"]["attribute"] == "blue"
    assert c.post("/api/observations/zzz/review", json={"reviewer_id": "r", "action": "accept"}).status_code == 404
    assert c.post("/api/observations/o1/review", json={"reviewer_id": "r", "action": "identify"}).status_code == 400


def test_narrative_and_group(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    n = c.post("/api/narrative", json={"text": "I think he wore a blue shirt."}).json()
    assert n["observations"][0]["uncertainty"]["linguistic_certainty"] == "hedged"
    g = c.post("/api/groups/aggregate", json={"text": "He wore a blue shirt.", "uids": ["coco_000000000139"]}).json()
    assert g["narrative_support"][0]["status"] == "attribute_mismatch"


def test_report(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    md = c.get("/api/report/coco_000000000139").text
    assert "no identity determination" in md and "shirt" in md
    assert "<table" in c.get("/api/report/coco_000000000139", params={"format": "html"}).text


def test_batch_job_mock(tmp_path, monkeypatch):
    c, appmod = _client(tmp_path, monkeypatch)
    from vle.runio import ROOT
    if not (ROOT / "data/images/coco/coco_000000000139.jpg").exists():
        return  # images not downloaded in this environment
    j = c.post("/api/batch", json={"uids": ["coco_000000000139"], "config": "configs/cpu_smoke.yaml"}).json()
    for _ in range(50):
        s = c.get(f"/api/jobs/{j['job_id']}").json()
        if s["status"] != "running":
            break
        time.sleep(0.1)
    assert s["status"] == "finished" and s["stats"]["n_ok"] == 1
