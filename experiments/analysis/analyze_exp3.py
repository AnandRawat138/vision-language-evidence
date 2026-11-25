"""Experiment 3: multi-image contextual reasoning vs independent / batched / aggregated descriptions.

Groups: one COCO/VQA test image -> 4 synthetic 'captures' of the same scene
  v0 left 70% crop, v1 right 70% crop, v2 full frame low-light (gamma 2.2 + noise), v3 full frame Gaussian blur (sigma 2.5).
Group reference = COCO-80 objects of the full image (exhaustive). Per-view visibility reference =
objects with >=50% of their box area inside the view (crops) or all objects (full-frame views).

Conditions
  C1 independent per-view descriptions (B=1)          -> mean per-view metrics, union, inter-view agreement
  C2 batched independent per-view descriptions (B=16) -> all groups; C1 = B=1 on a 30-group subset
  C3 contextual multi-image reasoning (4 images, one request)
  C4 C3 + structured aggregation (contextual labels verified by quality/score-weighted independent-view support)
  A  structured aggregation of independent views alone (Module F)
Thresholds (support fraction) tuned on the 40 dev groups.

    python experiments/analysis/analyze_exp3.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from run_inference import VIEW_SPECS, make_views, subset_rows  # noqa: E402
from vle.aggregation.aggregator import AggregationConfig, aggregate, kept_label_sets  # noqa: E402
from vle.evaluation import pipeline as P  # noqa: E402
from vle.evaluation.metrics import jaccard, prf  # noqa: E402
from vle.lexicon import map_term  # noqa: E402
from vle.parsing import parse_structured  # noqa: E402
from vle.preprocessing.quality import assess_quality, load_rgb  # noqa: E402
from vle.reporting.tables import ci, esc, fmt, write_table  # noqa: E402
from vle.runio import RAW, read_jsonl, write_json  # noqa: E402
from vle.schemas import VLMStructuredOutput  # noqa: E402
from vle.stats import tests as S  # noqa: E402

PROC = ROOT / "results" / "processed"


class ContextOutput(VLMStructuredOutput):
    pass


def visible_in_crop(box, img_w, img_h, which, frac=0.7) -> bool:  # type: ignore[no-untyped-def]
    cw, ch = int(img_w * frac), int(img_h * frac)
    x0, y0 = {"left": (0, (img_h - ch) // 2), "right": (img_w - cw, (img_h - ch) // 2)}[which]
    x, y, w, h = box
    ix = max(0, min(x + w, x0 + cw) - max(x, x0))
    iy = max(0, min(y + h, y0 + ch) - max(y, y0))
    return w * h > 0 and ix * iy / (w * h) >= 0.5


def view_refs(ref: dict, man_row: dict) -> list[set]:
    W, H = int(man_row["width"]), int(man_row["height"])
    out = []
    for which, _deg in VIEW_SPECS:
        if which in ("left", "right"):
            out.append({b["category"] for b in ref["object_boxes"] if visible_in_crop(b["bbox"], W, H, which)})
        else:
            out.append(set(ref["objects"]))
    return out


def f1(pred: set, ref: set) -> float:
    if not pred and not ref:
        return 1.0
    return prf(len(pred & ref), len(pred - ref), len(ref - pred))[2]


def hall(pred: set, ref: set) -> float:
    return len(pred - ref) / len(pred) if pred else 0.0


def main() -> None:
    man = P.manifest_index()
    groups = {"dev": [r["uid"] for r in subset_rows("groups_dev")], "test": [r["uid"] for r in subset_rows("groups_test")]}
    refs = P.references(groups["dev"] + groups["test"])
    import pickle
    comb = pickle.loads((PROC / "combiner.pkl").read_bytes())["combined"]

    def view_obs(run_name: str) -> dict[str, list]:
        run = RAW / "exp3" / run_name
        if not (run / "outputs.jsonl").exists():
            return {}
        obs, info = P.run_observations(run, None, key="image_id")
        # quality flags per view + CLIP grounding on the view image + calibrated score
        qcache = {}
        imgs = {}
        for k in obs:
            uid, vi = k.split("#v")
            base = load_rgb(ROOT / man[uid]["local_path"])
            v = make_views(base, VIEW_SPECS[int(vi)])
            imgs[k] = v
            qcache[k] = assess_quality(v).flags
        for k, v in obs.items():
            for o in v:
                o.uncertainty.image_quality_flags = qcache[k]
        P.add_grounding(obs, imgs, f"exp3_{run_name}")
        for k, v in obs.items():
            if v:
                for o, s in zip(v, comb.predict([P.feature_row(o) for o in v])):
                    o.uncertainty.combined_score = float(s)
                    o.uncertainty.calibrated = True
        return obs

    indep = view_obs("views_batched_b16")     # per-view independent requests, executed in batches of 16
    b1 = view_obs("views_indep_b1")            # same requests executed one at a time (30-group subset)
    ctx_rows = read_jsonl(RAW / "exp3" / "views_context" / "outputs.jsonl")
    ctx: dict[str, dict] = {}
    ctx_attr: dict[str, list] = {}
    ctx_parse = defaultdict(int)
    for r in ctx_rows:
        uid = r["meta"]["uid"]
        parsed, status, _ = parse_structured(r["text"])
        ctx_parse[status] += 1
        labels: set = set()
        attrib = []
        if parsed is not None:
            raw = json.loads(json.dumps(parsed.model_dump()))
            for it in parsed.objects:
                c = map_term(it.name, "coco")
                if c:
                    labels.add(c)
            # image attribution (field 'images' is ignored by the Pydantic model; re-read raw JSON)
            try:
                from vle.parsing import _extract_object, _repair
                rawd = json.loads(_repair(_extract_object(r["text"]) or "{}"))
                for it in rawd.get("objects", []) or []:
                    if isinstance(it, dict):
                        c = map_term(str(it.get("name", "")), "coco")
                        ims = it.get("images")
                        if c and isinstance(ims, list):
                            attrib.append((c, {int(i) - 1 for i in ims if str(i).isdigit()}))
            except Exception:  # noqa: BLE001
                pass
            del raw
        ctx[uid] = {"object": labels}
        ctx_attr[uid] = attrib

    def per_view_sets(obs: dict, uid: str) -> list[set]:
        return [{o.canonical for o in obs.get(f"{uid}#v{i}", []) if o.category.value == "object" and o.canonical}
                for i in range(len(VIEW_SPECS))]

    def agg_sets(obs: dict, uid: str, frac: float) -> set:
        per = {f"{uid}#v{i}": [o for o in obs.get(f"{uid}#v{i}", []) if o.category.value == "object"] for i in range(len(VIEW_SPECS))}
        a = aggregate(uid, per, AggregationConfig(min_support_frac=frac, use_quality_weights=True, use_scores=True))
        return kept_label_sets(a)["object"]

    def agg_support(obs: dict, uid: str) -> dict[str, float]:
        per = {f"{uid}#v{i}": [o for o in obs.get(f"{uid}#v{i}", []) if o.category.value == "object"] for i in range(len(VIEW_SPECS))}
        a = aggregate(uid, per, AggregationConfig(min_support_frac=0.0))
        return {d["label"]: d["weighted_support"] for d in a.consolidated}

    # ---- tune support thresholds on dev
    def tune(fn) -> float:  # type: ignore[no-untyped-def]
        best = (-1, 0.0)
        for t in np.arange(0.0, 1.01, 0.05):
            s = np.mean([f1(fn(u, t), set(refs[u]["objects"])) for u in groups["dev"] if u in ctx or indep])
            if s > best[0]:
                best = (s, float(t))
        return best[1]

    t_agg = tune(lambda u, t: agg_sets(indep, u, t)) if indep else 0.5
    t_ctx = tune(lambda u, t: {l for l in ctx.get(u, {}).get("object", set()) if agg_support(indep, u).get(l, 0) >= t}) if indep and ctx else 0.0

    test = [u for u in groups["test"] if u in ctx and any(f"{u}#v0" in d for d in (indep,))]
    m: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    identical = []
    for u in test:
        g = set(refs[u]["objects"])
        vr = view_refs(refs[u], man[u])
        pv = per_view_sets(indep, u)
        conds = {
            "C2 batched independent (per view, mean)": None,
            "C2 independent views (union)": set().union(*pv),
            "A  independent + structured aggregation": agg_sets(indep, u, t_agg),
            "C3 contextual multi-image": ctx[u]["object"],
            "C4 contextual + structured aggregation": {l for l in ctx[u]["object"] if agg_support(indep, u).get(l, 0) >= t_ctx},
        }
        if b1 and f"{u}#v0" in b1:
            pb = per_view_sets(b1, u)
            identical.extend(int(a == b) for a, b in zip(pv, pb))
            for nm_, sets_ in (("C1 independent B=1 (subset, per view)", pb), ("C2 batched B=16 (same subset, per view)", pv)):
                m[nm_]["f1"].append(np.mean([f1(s, g) for s in sets_]))
                m[nm_]["hall"].append(np.mean([hall(s, g) for s in sets_]))
                m[nm_]["agree"].append(np.mean([jaccard(sets_[i], sets_[j]) for i in range(4) for j in range(i + 1, 4)]))
        m["C2 batched independent (per view, mean)"]["f1"].append(np.mean([f1(s, g) for s in pv]))
        m["C2 batched independent (per view, mean)"]["view_f1"].append(np.mean([f1(s, r) for s, r in zip(pv, vr)]))
        m["C2 batched independent (per view, mean)"]["hall"].append(np.mean([hall(s, g) for s in pv]))
        m["C2 batched independent (per view, mean)"]["agree"].append(np.mean([jaccard(pv[i], pv[j]) for i in range(4) for j in range(i + 1, 4)]))
        for name, s in conds.items():
            if s is None:
                continue
            m[name]["f1"].append(f1(s, g))
            m[name]["hall"].append(hall(s, g))
            m[name]["recall"].append(len(s & g) / len(g) if g else 1.0)
        # contextual attribution accuracy vs per-view visibility
        att = ctx_attr.get(u, [])
        for lab, ims in att:
            truth = {i for i in range(4) if lab in vr[i]}
            if lab in g:
                m["C3 contextual multi-image"]["attrib_jacc"].append(jaccard(ims, truth))

    # transparency sweep (NOT used for tuning): aggregation threshold vs precision/recall/hallucination on test groups,
    # with and without image-quality weighting
    sweep = []
    for weighted in (True, False):
        for th in (0.0, 0.25, 0.5, 0.75):
            P_ = R_ = H_ = F_ = 0.0
            for u in test:
                g = set(refs[u]["objects"])
                per = {f"{u}#v{i}": [o for o in indep.get(f"{u}#v{i}", []) if o.category.value == "object"] for i in range(4)}
                a = aggregate(u, per, AggregationConfig(min_support_frac=th, use_quality_weights=weighted, use_scores=True))
                s = kept_label_sets(a)["object"]
                P_ += len(s & g) / len(s) if s else 1.0
                R_ += len(s & g) / len(g) if g else 1.0
                H_ += hall(s, g)
                F_ += f1(s, g)
            n = len(test)
            sweep.append({"theta": th, "quality_weighted": weighted, "precision": P_ / n, "recall": R_ / n,
                          "hallucination": H_ / n, "f1": F_ / n})
    out_sweep = sweep
    out = {"n_groups_test": len(test), "aggregation_sweep_test": out_sweep, "t_agg_dev": t_agg, "t_ctx_dev": t_ctx, "context_parse": dict(ctx_parse),
           "batched_identical_view_outputs": float(np.mean(identical)) if identical else None, "conditions": {}}
    for name, d in m.items():
        out["conditions"][name] = {k: S.describe(v) for k, v in d.items()}
    # paired tests on group-level F1 and hallucination
    ref_name = "C3 contextual multi-image"
    pv = {}
    for name in ["C2 independent views (union)", "A  independent + structured aggregation", "C4 contextual + structured aggregation",
                 "C2 batched independent (per view, mean)"]:
        if name in m and m[name]["f1"]:
            for metric in ("f1", "hall"):
                a, b = np.array(m[name][metric]), np.array(m[ref_name][metric])
                key = f"{name} vs C3 [{metric}]"
                pv[key] = {**S.paired_bootstrap(a, b), "wilcoxon_p": S.wilcoxon(a, b)["p"], "cohens_dz": S.cohens_dz(a, b)}
    a = np.array(m["A  independent + structured aggregation"]["f1"])
    b = np.array(m["C2 batched independent (per view, mean)"]["f1"])
    pv["A vs C1 per-view [f1]"] = {**S.paired_bootstrap(a, b), "wilcoxon_p": S.wilcoxon(a, b)["p"], "cohens_dz": S.cohens_dz(a, b)}
    a = np.array(m["A  independent + structured aggregation"]["hall"])
    b = np.array(m["C2 batched independent (per view, mean)"]["hall"])
    pv["A vs C1 per-view [hall]"] = {**S.paired_bootstrap(a, b), "wilcoxon_p": S.wilcoxon(a, b)["p"], "cohens_dz": S.cohens_dz(a, b)}
    adj = S.holm({k: v["wilcoxon_p"] for k, v in pv.items()})
    for k in pv:
        pv[k]["wilcoxon_p_holm"] = adj[k]
    out["paired_tests"] = pv
    write_json(PROC / "exp3_summary.json", out)

    nice = {"C2 batched independent (per view, mean)": "Independent, single view (mean of 4)",
            "A  independent + structured aggregation": f"Aggregation, dev-tuned $\\theta={t_agg:.2f}$ (= union)",
            "C3 contextual multi-image": "Contextual 4-image prompt (C3)",
            "C4 contextual + structured aggregation": f"C3 verified by aggregation ($\\theta={t_ctx:.2f}$)"}
    rows = []
    for name in nice:
        if name not in out["conditions"]:
            continue
        d = out["conditions"][name]
        g = lambda k: ci(d[k]["mean"], d[k]["ci95_lo"], d[k]["ci95_hi"]) if k in d and d[k].get("n") else "--"  # noqa: E731
        rows.append([nice[name], g("f1"), g("hall"), g("agree")])
    rows.append(["MIDRULE"])
    for r in out_sweep:
        if r["theta"] in (0.25, 0.5):
            rows.append([f"Aggregation $\\theta={r['theta']:.2f}$, {'quality-weighted' if r['quality_weighted'] else 'unweighted'}$^\\dagger$",
                         fmt(r["f1"]), fmt(r["hallucination"]), "--"])
    write_table("exp3_context", ["Condition", "Group F1", "Hallucination rate", "Inter-view Jaccard"], rows,
                f"Multi-image consolidation on {len(test)} test groups (four cropped/degraded captures of one scene; COCO-80 "
                "objects of the full image as reference). Single-view values are averaged over views; inter-view Jaccard "
                "measures agreement between the four independent outputs. 95\\% bootstrap CIs. $^\\dagger$Threshold sweep on the "
                "test groups, reported for transparency and not used for tuning.",
                "tab:exp3", colspec="lccc", size=r"\footnotesize", wide=False)
    print(json.dumps({k: {kk: vv.get("mean") for kk, vv in v.items()} for k, v in out["conditions"].items()}, indent=1))
    print("identical", out["batched_identical_view_outputs"], "t", t_agg, t_ctx)


if __name__ == "__main__":
    main()
