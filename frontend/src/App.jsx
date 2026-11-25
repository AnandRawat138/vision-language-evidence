import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";

const TABS = ["Overview", "Images", "Batch", "Consistency", "Performance", "Audit"];

export default function App() {
  const [tab, setTab] = useState("Overview");
  const [uid, setUid] = useState(() => {
    const m = window.location.hash.match(/^#\/image\/(.+)$/);
    return m ? decodeURIComponent(m[1]) : null;
  });
  const [reviewer, setReviewer] = useState(() => {
    try { return localStorage.getItem("vle.reviewer") || ""; } catch { return ""; }
  });
  useEffect(() => { try { localStorage.setItem("vle.reviewer", reviewer); } catch {} }, [reviewer]);
  return (
    <div className="app">
      <header>
        <h1>Evidence Review</h1>
        <nav>{TABS.map((t) => (
          <button key={t} className={t === tab && !uid ? "on" : ""} onClick={() => { setTab(t); setUid(null); }}>{t}</button>
        ))}</nav>
        <label className="reviewer">Reviewer ID <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="required to review" /></label>
      </header>
      <div className="notice">Observable content only. No identity matching, face recognition or person ranking. Scores are uncalibrated unless marked calibrated.</div>
      <main>
        {uid ? <ImageDetail uid={uid} reviewer={reviewer} onBack={() => setUid(null)} /> :
          tab === "Overview" ? <Overview /> :
          tab === "Images" ? <Browser onOpen={setUid} /> :
          tab === "Batch" ? <Batch /> :
          tab === "Consistency" ? <Consistency /> :
          tab === "Performance" ? <Performance /> : <Audit onOpen={setUid} />}
      </main>
    </div>
  );
}

function useApi(path, deps = []) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const reload = () => api(path).then(setD).catch((e) => setErr(String(e)));
  useEffect(() => { reload(); }, deps);
  return [d, err, reload];
}

function Overview() {
  const [d, err] = useApi("/overview");
  const [m] = useApi("/metrics");
  if (err) return <p className="err">{err}</p>;
  if (!d) return <p>Loading…</p>;
  return (
    <section>
      <h2>Dataset</h2>
      <table><thead><tr><th>Source</th><th>Split</th><th>Images</th></tr></thead>
        <tbody>{d.images.by_source_split.map((r) => <tr key={r.source + r.split}><td>{r.source}</td><td>{r.split}</td><td>{r.n}</td></tr>)}</tbody>
        <tfoot><tr><td colSpan="2">Total</td><td>{d.images.total}</td></tr></tfoot></table>
      <h2>Runs in store</h2><p>{d.runs.join(", ") || "none"}</p>
      <h2>Review status</h2>
      <table><tbody>{Object.entries(d.review.status).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}</tbody></table>
      {m && Object.keys(m).length > 0 && <><h2>Experiment summaries</h2>
        {Object.entries(m).map(([k, v]) => <details key={k}><summary>{k}</summary><pre>{JSON.stringify(v, null, 1)}</pre></details>)}</>}
    </section>
  );
}

function Browser({ onOpen }) {
  const [source, setSource] = useState("");
  const [split, setSplit] = useState("");
  const [page, setPage] = useState(0);
  const q = `/images?limit=48&offset=${page * 48}${source ? `&source=${source}` : ""}${split ? `&split=${split}` : ""}`;
  const [rows, err] = useApi(q, [q]);
  return (
    <section>
      <div className="toolbar">
        <select value={source} onChange={(e) => { setSource(e.target.value); setPage(0); }}>
          <option value="">all sources</option>{["coco", "vqa", "vg", "flickr30k"].map((s) => <option key={s}>{s}</option>)}
        </select>
        <select value={split} onChange={(e) => { setSplit(e.target.value); setPage(0); }}>
          <option value="">all splits</option><option>dev</option><option>test</option>
        </select>
        <button disabled={page === 0} onClick={() => setPage(page - 1)}>‹ Prev</button>
        <span>page {page + 1}</span>
        <button onClick={() => setPage(page + 1)}>Next ›</button>
      </div>
      {err && <p className="err">{err}</p>}
      <div className="grid">{(rows || []).map((r) => (
        <figure key={r.uid} onClick={() => onOpen(r.uid)}>
          <img loading="lazy" src={`/api/images/${r.uid}/file`} alt={r.uid} />
          <figcaption>{r.uid}<br /><small>{r.source} · {r.split}</small></figcaption>
        </figure>))}</div>
    </section>
  );
}

function score(u) {
  const s = u.combined_score ?? u.token_confidence;
  return s == null ? "–" : s.toFixed(2);
}

function ImageDetail({ uid, reviewer, onBack }) {
  const [d, err, reload] = useApi(`/images/${uid}`, [uid]);
  const [showRef, setShowRef] = useState(false);
  const started = useRef(Date.now());
  useEffect(() => { started.current = Date.now(); }, [uid]);
  if (err) return <p className="err">{err}</p>;
  if (!d) return <p>Loading…</p>;
  const act = async (o, action, after) => {
    if (!reviewer) { alert("Enter a reviewer ID first."); return; }
    await api(`/observations/${o.observation_id}/review`, { method: "POST",
      body: { reviewer_id: reviewer, action, after, duration_ms: Date.now() - started.current } });
    started.current = Date.now();
    reload();
  };
  const info = Object.values(d.info || {})[0] || {};
  return (
    <section className="detail">
      <button onClick={onBack}>‹ Back</button>
      <h2>{uid} <small>{d.meta.source} / {d.meta.split}</small></h2>
      <div className="cols">
        <div>
          <img className="big" src={`/api/images/${uid}/file`} alt={uid} />
          <p><small>md5 {d.meta.md5} · {d.meta.width}×{d.meta.height}</small></p>
          {info.quality && <p>Quality flags: {(info.quality.flags || []).join(", ") || "none"} (Laplacian var {info.quality.blur_laplacian_var}, brightness {info.quality.brightness_mean})</p>}
          {info.parse_status && <p>Structured output: <b>{info.parse_status}</b></p>}
          {info.unclear?.length > 0 && <p>Model-reported unclear: {info.unclear.join("; ")}</p>}
          <p><a href={`/api/report/${uid}?format=html`} target="_blank">HTML report</a> · <a href={`/api/report/${uid}?format=md`} target="_blank">Markdown</a> · <a href={`/api/report/${uid}?format=json`} target="_blank">JSON</a></p>
        </div>
        <div>
          <h3>Model observations</h3>
          <table className="obs"><thead><tr><th>Cat.</th><th>Label</th><th>Raw</th><th>Attr</th><th>Score</th><th>Verbal</th><th>CLIP</th><th>Status</th><th></th></tr></thead>
            <tbody>{d.observations.map((o) => (
              <tr key={o.observation_id} className={o.uncertainty.needs_review ? "flag" : ""}>
                <td>{o.category}</td><td>{o.label}</td><td>{o.raw_text}</td><td>{o.attribute || ""}</td>
                <td title={o.uncertainty.calibrated ? "calibrated (dev-fit)" : "uncalibrated"}>{score(o.uncertainty)}{o.uncertainty.calibrated ? "ᶜ" : ""}</td>
                <td>{o.uncertainty.verbal_confidence || ""}</td><td>{o.uncertainty.grounding_score?.toFixed(3) ?? ""}</td>
                <td>{o.review_status}</td>
                <td className="actions">
                  <button onClick={() => act(o, "accept")}>✓</button>
                  <button onClick={() => act(o, "reject")}>✗</button>
                  <button onClick={() => { const v = prompt("Corrected label", o.label); if (v) act(o, "correct", { label: v, canonical: v }); }}>label</button>
                  {o.category === "clothing" && <button onClick={() => { const v = prompt("Corrected colour", o.attribute || ""); if (v != null) act(o, "correct", { attribute: v }); }}>colour</button>}
                </td>
              </tr>))}</tbody></table>
          {d.aggregation?.contradictions?.length > 0 && <><h3>Consistency issues</h3>
            <ul>{d.aggregation.contradictions.map((c, i) => <li key={i}><b>{c.slot}</b>: {c.explanation}</li>)}</ul></>}
          {d.aggregation && <p>Not observed: {d.aggregation.missing_information.join(", ") || "–"}</p>}
          <label><input type="checkbox" checked={showRef} onChange={(e) => setShowRef(e.target.checked)} /> show original dataset annotations (evaluation reference — not model output)</label>
          {showRef && d.reference && <pre className="ref">{JSON.stringify({ objects: d.reference.objects, clothing: d.reference.clothing, accessories: d.reference.accessories, carried: d.reference.carried, scene: d.reference.scene_places, captions: d.reference.captions }, null, 1)}</pre>}
          <h3>Audit trail</h3>
          <table><tbody>{d.audit.map((a) => <tr key={a.event_id}><td>{a.timestamp}</td><td>{a.reviewer_id}</td><td>{a.action}</td><td>{a.before?.label}→{a.after?.label} {a.after?.attribute}</td></tr>)}</tbody></table>
        </div>
      </div>
    </section>
  );
}

function Batch() {
  const [uids, setUids] = useState("");
  const [policy, setPolicy] = useState("static");
  const [bs, setBs] = useState(4);
  const [config, setConfig] = useState("configs/cpu_smoke.yaml");
  const [job, setJob] = useState(null);
  const [st, setSt] = useState(null);
  useEffect(() => {
    if (!job) return;
    const t = setInterval(() => api(`/jobs/${job}`).then((s) => { setSt(s); if (s.status !== "running") clearInterval(t); }), 800);
    return () => clearInterval(t);
  }, [job]);
  const fill = async () => setUids((await api("/images?limit=16&split=dev")).map((r) => r.uid).join("\n"));
  return (
    <section>
      <h2>Batch execution</h2>
      <p>Independent requests are scheduled in batches. Contextual multi-image reasoning is a different operation and is not run here.</p>
      <div className="toolbar">
        <select value={policy} onChange={(e) => setPolicy(e.target.value)}>{["independent", "static", "sorted", "adaptive"].map((p) => <option key={p}>{p}</option>)}</select>
        <label>batch <input type="number" min="1" max="64" value={bs} onChange={(e) => setBs(+e.target.value)} /></label>
        <select value={config} onChange={(e) => setConfig(e.target.value)}><option>configs/cpu_smoke.yaml</option><option>configs/local_gpu.yaml</option></select>
        <button onClick={fill}>Fill 16 dev images</button>
        <button onClick={async () => setJob((await api("/batch", { method: "POST", body: { uids: uids.split(/\s+/).filter(Boolean), policy, batch_size: bs, config } })).job_id)}>Run</button>
      </div>
      <textarea rows="8" value={uids} onChange={(e) => setUids(e.target.value)} placeholder="image uids, one per line" />
      {st && <pre>{JSON.stringify(st, null, 1)}</pre>}
    </section>
  );
}

function Consistency() {
  const [uids, setUids] = useState("");
  const [text, setText] = useState("");
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  const run = () => api("/groups/aggregate", { method: "POST", body: { uids: uids.split(/\s+/).filter(Boolean), text } }).then(setRes).catch((e) => setErr(String(e)));
  return (
    <section>
      <h2>Cross-image consistency</h2>
      <p>Summarises how many selected images support each observation and where they disagree. It never asserts that images show the same person.</p>
      <textarea rows="4" value={uids} onChange={(e) => setUids(e.target.value)} placeholder="image uids" />
      <textarea rows="3" value={text} onChange={(e) => setText(e.target.value)} placeholder="optional witness-style statement (kept verbatim; hedged statements stay hedged)" />
      <button onClick={run}>Analyse</button>
      {err && <p className="err">{err}</p>}
      {res && <>
        <p><i>{res.disclaimer}</i></p>
        <h3>Consolidated observations</h3>
        <table><thead><tr><th>Category</th><th>Label</th><th>Support</th><th>Weighted</th><th>Images</th></tr></thead>
          <tbody>{res.consolidated.map((c) => <tr key={c.category + c.label}><td>{c.category}</td><td>{c.label}</td><td>{c.n_support}/{res.image_ids.length}</td><td>{c.weighted_support}</td><td>{c.support_images.join(", ")}</td></tr>)}</tbody></table>
        <h3>Contradictions</h3>
        <ul>{res.contradictions.map((c, i) => <li key={i}><b>{c.slot}</b> [{c.kind}, {c.severity}] {c.explanation} — {Object.keys(c.values).join(" vs ")}</li>)}</ul>
        {res.narrative_support.length > 0 && <><h3>Statement vs images</h3>
          <table><thead><tr><th>Item</th><th>Attr</th><th>Certainty</th><th>Status</th><th>Images</th></tr></thead>
            <tbody>{res.narrative_support.map((n) => <tr key={n.observation_id}><td>{n.label}</td><td>{n.attribute || ""}</td><td>{n.certainty}</td><td>{n.status}</td><td>{n.images.join(", ")}</td></tr>)}</tbody></table></>}
        <p>Missing: {res.missing_information.join(", ") || "–"}</p>
      </>}
    </section>
  );
}

function Performance() {
  const [d, err] = useApi("/performance");
  if (err) return <p className="err">{err}</p>;
  if (!d) return <p>Loading…</p>;
  if (!d.rows) return <pre>{JSON.stringify(d, null, 1)}</pre>;
  const max = Math.max(...d.rows.map((r) => r.throughput_mean || 0));
  return (
    <section>
      <h2>Batch processing performance (Experiment 2)</h2>
      <table><thead><tr><th>Policy</th><th>Batch</th><th>img/s (mean±sd)</th><th>p50 s</th><th>p95 s</th><th>GPU MB</th><th></th></tr></thead>
        <tbody>{d.rows.map((r) => <tr key={r.policy + r.batch}>
          <td>{r.policy}</td><td>{r.batch}</td><td>{r.throughput_mean?.toFixed(3)} ± {r.throughput_sd?.toFixed(3)}</td>
          <td>{r.latency_p50?.toFixed(2)}</td><td>{r.latency_p95?.toFixed(2)}</td><td>{r.gpu_mem_peak_mb?.toFixed(0)}</td>
          <td><div className="bar" style={{ width: `${(100 * (r.throughput_mean || 0)) / max}%` }} /></td></tr>)}</tbody></table>
    </section>
  );
}

function Audit({ onOpen }) {
  const [d, err] = useApi("/audit");
  if (err) return <p className="err">{err}</p>;
  return (
    <section>
      <h2>Audit trail (all corrections)</h2>
      <table><thead><tr><th>Time</th><th>Reviewer</th><th>Action</th><th>Image</th><th>Before</th><th>After</th><th>ms</th></tr></thead>
        <tbody>{(d || []).map((a) => <tr key={a.event_id}><td>{a.timestamp}</td><td>{a.reviewer_id}</td><td>{a.action}</td>
          <td><a href="#" onClick={(e) => { e.preventDefault(); onOpen(a.image_id); }}>{a.image_id}</a></td>
          <td>{a.before?.label} {a.before?.attribute}</td><td>{a.after?.label} {a.after?.attribute}</td><td>{a.duration_ms}</td></tr>)}</tbody></table>
    </section>
  );
}
