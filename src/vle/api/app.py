"""Module H backend: FastAPI service for the human-review dashboard.

    uvicorn vle.api.app:app --port 8008      (VLE_DB, VLE_BACKEND_CONFIG env vars)

Only observable-content functionality is exposed. There are no endpoints for
identity search, face matching, person ranking or line-up generation.
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vle.aggregation.aggregator import AggregationConfig, aggregate
from vle.api.store import EvidenceStore
from vle.datasets.reference import load_reference
from vle.narrative.structurer import RuleStructurer
from vle.reporting.report import image_report_markdown, markdown_to_html
from vle.runio import ROOT
from vle.schemas import CorrectionEvent

DB_PATH = os.environ.get("VLE_DB", str(ROOT / "results" / "evidence.sqlite"))
app = FastAPI(title="VLE Evidence Review API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
store = EvidenceStore(DB_PATH)
JOBS: dict[str, dict] = {}


class ReviewIn(BaseModel):
    reviewer_id: str
    action: str
    after: dict | None = None
    comment: str = ""
    duration_ms: int | None = None


class BatchIn(BaseModel):
    uids: list[str]
    policy: str = "static"
    batch_size: int = 4
    config: str = "configs/cpu_smoke.yaml"


class NarrativeIn(BaseModel):
    text: str
    uids: list[str] = []


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "db": DB_PATH}


@app.get("/api/overview")
def overview() -> dict:
    return {"images": store.count_images(), "runs": store.runs(), "review": store.review_stats()}


@app.get("/api/images")
def images(source: str | None = None, split: str | None = None, offset: int = 0, limit: int = Query(50, le=500)) -> list[dict]:
    return store.images(source, split, offset, limit)


@app.get("/api/images/{uid}")
def image(uid: str, run_id: str | None = None) -> dict:
    meta = store.image(uid)
    if meta is None:
        raise HTTPException(404, "unknown image")
    try:
        ref = load_reference(ROOT, uid)
    except FileNotFoundError:
        ref = None
    obs = store.observations(uid, run_id)
    agg = aggregate(uid, {uid: obs}, AggregationConfig(min_support_frac=0.0)) if obs else None
    return {"meta": meta, "reference": ref, "observations": [o.model_dump(mode="json") for o in obs],
            "info": store.image_info(uid), "aggregation": agg.model_dump(mode="json") if agg else None,
            "audit": store.audit(uid)}


@app.get("/api/images/{uid}/file")
def image_file(uid: str) -> FileResponse:
    meta = store.image(uid)
    if meta is None:
        raise HTTPException(404)
    p = (ROOT / meta["local_path"]).resolve()
    if ROOT.resolve() not in p.parents:
        raise HTTPException(400)
    return FileResponse(p)


@app.post("/api/observations/{observation_id}/review")
def review(observation_id: str, body: ReviewIn) -> dict:
    if body.action not in {"accept", "reject", "correct", "add"}:
        raise HTTPException(400, "invalid action")
    ev = CorrectionEvent(observation_id=observation_id, reviewer_id=body.reviewer_id, action=body.action,  # type: ignore[arg-type]
                         after=body.after, comment=body.comment, duration_ms=body.duration_ms)
    try:
        o = store.review(ev)
    except KeyError:
        raise HTTPException(404, "unknown observation") from None
    return {"observation": o.model_dump(mode="json"), "event_id": ev.event_id}


@app.get("/api/audit")
def audit(image_id: str | None = None) -> list[dict]:
    return store.audit(image_id)


@app.post("/api/groups/aggregate")
def group_aggregate(body: NarrativeIn) -> dict:
    """Consistency analysis over a user-selected set of images, optionally against a statement."""
    per = {u: store.observations(u) for u in body.uids}
    narr = RuleStructurer().structure(body.text).observations if body.text.strip() else None
    return aggregate("adhoc-" + uuid.uuid4().hex[:6], per, narrative=narr).model_dump(mode="json")


@app.post("/api/narrative")
def narrative(body: NarrativeIn) -> dict:
    r = RuleStructurer().structure(body.text)
    return {"original_text": r.original_text, "observations": [o.model_dump(mode="json") for o in r.observations],
            "missing": r.missing, "ambiguous": r.ambiguous_spans}


@app.get("/api/performance")
def performance() -> dict:
    p = ROOT / "results" / "processed" / "exp2_summary.json"
    return json.loads(p.read_text()) if p.exists() else {"status": "not available"}


@app.get("/api/metrics")
def metrics() -> dict:
    out = {}
    for f in glob.glob(str(ROOT / "results" / "processed" / "*_summary.json")):
        out[Path(f).stem] = json.loads(Path(f).read_text())
    return out


@app.post("/api/batch")
def batch(body: BatchIn) -> dict:
    """Launch a batch job in a background thread (mock backend by default for the demo UI)."""
    import yaml

    from vle.batching.scheduler import BatchScheduler, SchedulerConfig
    from vle.inference import make_backend
    from vle.inference.base import InferenceRequest
    from vle.preprocessing.quality import load_rgb
    from vle.prompts import get_prompt, render
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "n": len(body.uids), "done": 0, "started": time.time()}

    def work() -> None:
        try:
            cfg = yaml.safe_load((ROOT / body.config).read_text())
            be = make_backend(cfg)
            p = get_prompt("struct-v1")
            reqs = []
            for u in body.uids:
                m = store.image(u)
                if m:
                    reqs.append(InferenceRequest(prompt_id="struct-v1", text=render("struct-v1"), system=p["system"],
                                                 images=[load_rgb(ROOT / m["local_path"])], image_ids=[u]))

            def cb(req, res):  # type: ignore[no-untyped-def]
                JOBS[job_id]["done"] += 1
            st = BatchScheduler(be, SchedulerConfig(policy=body.policy, batch_size=body.batch_size), on_result=cb).run(reqs)
            JOBS[job_id].update({"status": "finished", "stats": st.summary(), "backend": be.info()})
        except Exception as e:  # noqa: BLE001
            JOBS[job_id].update({"status": "failed", "error": str(e)})
    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404)
    return JOBS[job_id]


@app.get("/api/report/{uid}")
def report(uid: str, format: str = "md"):  # type: ignore[no-untyped-def]
    data = image(uid)
    md = image_report_markdown(data)
    if format == "html":
        return HTMLResponse(markdown_to_html(md))
    if format == "json":
        return data
    return PlainTextResponse(md, media_type="text/markdown")


_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="ui")
