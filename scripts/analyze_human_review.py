"""Analyse returned blind-review files (real reviewers only). Compares answers with dataset references and calibrated scores."""
import collections, json, statistics as st, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
from scipy.stats import beta
from vle.datasets.reference import load_reference
R = lambda u: set(load_reference(ROOT, u)["objects"])
obs = {o["observation_id"]: o for o in map(json.loads, open(ROOT/"results/processed/exp1_observations.jsonl", encoding="utf8"))}
def cp(k, n): return [beta.ppf(.025, k, n-k+1) if k else 0., beta.ppf(.975, k+1, n-k) if k < n else 1.]
out = {}
for f in sorted((ROOT/"results/raw/human_review").glob("review_*.json")):
    d = json.load(open(f, encoding="utf8")); A = d["answers"]; tab = collections.Counter(); dis = []
    assert len({a["oid"] for a in A}) == len(A)
    for a in A:
        o = obs[a["oid"]]; assert o["canonical"] == a["label"]
        inref = a["label"] in R(a["uid"]); tab[(a["answer"], inref)] += 1
        if (a["answer"] == "yes") != inref: dis.append([a["uid"], a["label"], a["answer"], inref, a["ms"]])
    n = len(A); y = sum(a["answer"] == "yes" for a in A); r = sum(v for k, v in tab.items() if k[1])
    out[f.stem] = {"author": d["author"], "n": n, "images": len({a["uid"] for a in A}), "answers": dict(collections.Counter(a["answer"] for a in A)),
        "visible_frac": y/n, "visible_ci": cp(y, n), "in_reference": r, "table": {f"{k[0]}|inref={k[1]}": v for k, v in tab.items()},
        "agree": n-len(dis), "disagreements": dis, "median_ms": st.median(a["ms"] for a in A), "min_ms": min(a["ms"] for a in A), "started": d["started"], "ended": d["ended"]}
(ROOT/"results/processed/human_review.json").write_text(json.dumps(out, indent=1), encoding="utf8"); print(json.dumps(out, indent=1))
