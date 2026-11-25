"""Experiments 1 (image understanding), 6 (uncertainty calibration) and parts of 7 (ablations).

Inputs (must exist):
  results/raw/exp1/struct_b16/       framework structured extraction, all 1,000 images
  results/raw/exp1/caption_b16/      same VLM, free-form captions (baseline B3)
  results/raw/baselines/b1_b2/      BLIP captions (B1) and CLIP scores (B2)
Protocol: every threshold / calibrator is fit on the DEV split; all reported numbers are on TEST.

    python experiments/analysis/analyze_exp1.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from vle.evaluation import pipeline as P  # noqa: E402
from vle.evaluation.metrics import prf  # noqa: E402
from vle.preprocessing.quality import load_rgb  # noqa: E402
from vle.reporting.tables import ci, esc, fmt, write_table  # noqa: E402
from vle.runio import RAW, write_json  # noqa: E402
from vle.stats import tests as S  # noqa: E402
from vle.uncertainty.calibration import (Combiner, auroc, brier, ece, reliability, risk_coverage,  # noqa: E402
                                         selective_risk_at)

PROC = ROOT / "results" / "processed"
# selection thresholds apply only to categories the calibrator was trained on (exhaustive/attribute references)
CALIBRATED = {"object", "clothing"}
CATS = [("objects", "object"), ("clothing", "clothing"), ("accessory", "accessory"), ("carried", "carried_object")]


# ------------------------------------------------------------------ helpers
def micro_ci(m: dict, refs_ids: list[str], preds: dict, refs: dict, cat_key: str, ref_key: str) -> dict:
    """Bootstrap CIs (image resampling) for micro P/R/F1."""
    tp, fp, fn = [], [], []
    for u in refs_ids:
        pr = preds.get(u, {}).get(cat_key, set())
        rf = set(refs[u][ref_key]) if not isinstance(refs[u][ref_key], dict) else set(refs[u][ref_key])
        tp.append(len(pr & rf))
        fp.append(len(pr - rf))
        fn.append(len(rf - pr))
    tp, fp, fn = map(np.array, (tp, fp, fn))
    rng = np.random.default_rng(S.SEED)
    idx = rng.integers(0, len(tp), size=(S.DEFAULT_B, len(tp)))
    T, F, N = tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1)
    P_ = T / np.maximum(T + F, 1)
    R_ = T / np.maximum(T + N, 1)
    F1 = 2 * P_ * R_ / np.maximum(P_ + R_, 1e-12)
    p, r, f = prf(tp.sum(), fp.sum(), fn.sum())
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return {"p": p, "p_ci": q(P_), "r": r, "r_ci": q(R_), "f1": f, "f1_ci": q(F1), "n_images": len(tp)}


def chair_ci(preds: dict, refs: dict, ids: list[str]) -> dict:
    h = np.array([len(preds.get(u, {}).get("object", set()) - set(refs[u]["objects"])) for u in ids])
    m = np.array([len(preds.get(u, {}).get("object", set())) for u in ids])
    ci_i = S.bootstrap_ratio_ci(h, m)
    s = (h > 0).astype(float)
    lo, hi = S.bootstrap_ci(s)
    return {"chair_i": ci_i[0], "chair_i_ci": ci_i[1:], "chair_s": float(s.mean()), "chair_s_ci": (lo, hi)}


def per_image_f1(preds: dict, refs: dict, ids: list[str], cat_key: str, ref_key: str) -> np.ndarray:
    out = []
    for u in ids:
        pr = preds.get(u, {}).get(cat_key, set())
        rf = set(refs[u][ref_key])
        if not pr and not rf:
            out.append(1.0)
        else:
            out.append(prf(len(pr & rf), len(pr - rf), len(rf - pr))[2])
    return np.array(out)


def full_eval(preds: dict, refs: dict, test: list[str]) -> dict:
    ou = [u for u in test if refs[u]["objects_complete"]]
    au = [u for u in test if refs[u].get("attr_available")]
    su = [u for u in test if refs[u]["scene_places"]]
    ev = P.evaluate_predictions(preds, refs, test)
    res = {"objects": micro_ci(ev["objects"], ou, preds, refs, "object", "objects"),
           "objects_macro_f1": ev["objects"]["macro_f1"],
           "chair": chair_ci(preds, refs, ou),
           "clothing": micro_ci(ev["clothing"], au, preds, refs, "clothing", "clothing"),
           "clothing_macro_f1": ev["clothing"]["macro_f1"],
           "accessory": micro_ci(ev["accessory"], au, preds, refs, "accessory", "accessories"),
           "carried": micro_ci(ev["carried"], au, preds, refs, "carried_object", "carried"),
           "color": ev["color"],
           "scene": {"hit_rate": ev["scene"]["hit_rate"], "n": ev["scene"]["n_images"],
                     "ci": S.bootstrap_ci(np.array(ev["scene"]["per_image"], float)) if su else None}}
    res["_per_image"] = {"objects": per_image_f1(preds, refs, ou, "object", "objects").tolist(),
                         "clothing": per_image_f1(preds, refs, au, "clothing", "clothing").tolist(),
                         "objects_ids": ou, "clothing_ids": au}
    return res


def clip_preds(scores: dict, space: dict, thr: dict[str, float], uids: list[str]) -> dict:
    out = {}
    for u in uids:
        d: dict[str, set] = defaultdict(set)
        for cat, labels in space.items():
            for l in labels:
                if scores[u][l] >= thr[cat]:
                    d[cat].add(l)
        d["clothing_colors"] = {}  # type: ignore[assignment]
        out[u] = d
    return out


def tune_clip(scores, space, refs, dev):  # type: ignore[no-untyped-def]
    """Per-category threshold maximising dev micro-F1 (objects: coco/vqa refs; attributes: vg/flickr refs)."""
    thr = {}
    keymap = {"object": ("objects", "objects_complete"), "clothing": ("clothing", "attr_available"),
              "accessory": ("accessories", "attr_available"), "carried_object": ("carried", "attr_available")}
    for cat, labels in space.items():
        rk, flag = keymap[cat]
        ids = [u for u in dev if refs[u].get(flag)]
        best = (-1, 0.25)
        for t in np.arange(0.18, 0.36, 0.0025):
            T = F = N = 0
            for u in ids:
                pr = {l for l in labels if scores[u][l] >= t}
                rf = set(refs[u][rk])
                T, F, N = T + len(pr & rf), F + len(pr - rf), N + len(rf - pr)
            f = prf(T, F, N)[2]
            if f > best[0]:
                best = (f, float(t))
        thr[cat] = best[1]
    return thr


def cluster_bootstrap_metric(rows: list[dict], conf_key: str, fn, B: int = 2000) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    by_img: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["uid"]].append(i)
    keys = list(by_img)
    conf = np.array([r[conf_key] for r in rows])
    y = np.array([r["y"] for r in rows])
    rng = np.random.default_rng(S.SEED)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_img[keys[k]] for k in pick])
        vals.append(fn(conf[idx], y[idx]))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------- main
def main() -> None:
    man = P.manifest_index()
    uids = sorted(man)
    dev = [u for u in uids if man[u]["split"] == "dev"]
    test = [u for u in uids if man[u]["split"] == "test"]
    refs = P.references(uids)
    quality = P.quality_cache(uids)

    # ---------------- framework structured outputs
    run = RAW / "exp1" / "struct_b16"
    obs, info = P.run_observations(run, quality)
    missing = [u for u in uids if u not in obs]
    images = {u: (lambda u=u: load_rgb(ROOT / man[u]["local_path"])) for u in uids}
    P.add_grounding(obs, images, "orig")
    rows = P.labelled_rows(obs, refs)
    dev_rows = [r for r in rows if man[r["uid"]]["split"] == "dev"]
    test_rows = [r for r in rows if man[r["uid"]]["split"] == "test"]
    y_dev = np.array([r["y"] for r in dev_rows])
    y_test = np.array([r["y"] for r in test_rows])

    # ---------------- calibrators (dev fit)
    variants = {
        "combined": Combiner(),
        "combined_no_ground": Combiner(features=["token", "verbal", "n_quality_flags", "is_object", "is_clothing"]),
        "combined_no_quality": Combiner(features=["token", "verbal", "ground", "is_object", "is_clothing"]),
        "token_platt": Combiner(features=["token"]),
        "ground_platt": Combiner(features=["ground"]),
        "verbal_platt": Combiner(features=["verbal"]),
    }
    for name, c in variants.items():
        c.fit(dev_rows, y_dev)
        for rr, p in zip(test_rows, c.predict(test_rows)):
            rr[name] = float(p)
        for rr, p in zip(dev_rows, c.predict(dev_rows)):
            rr[name] = float(p)
    for rr in test_rows + dev_rows:
        rr["token_raw"] = rr["token"] if rr["token"] is not None else 0.5
        rr["verbal_raw"] = rr["verbal"]

    # write calibrated score back onto every observation (test and dev, all categories with features)
    comb = variants["combined"]
    for k, v in obs.items():
        feats = [P.feature_row(o) for o in v]
        if feats:
            for o, s in zip(v, comb.predict(feats)):
                o.uncertainty.combined_score = float(s)
                o.uncertainty.calibrated = True

    # ---------------- selection threshold for the framework (dev, maximise object+clothing micro-F1)
    def preds_at(tau: float, keys: list[str]) -> dict:
        return P.pred_sets({k: obs.get(k, []) for k in keys},
                           keep=lambda o: (o.category.value not in CALIBRATED or o.uncertainty.combined_score is None
                                          or o.uncertainty.combined_score >= tau))

    best = (-1.0, 0.0)
    for tau in np.arange(0.0, 0.9, 0.02):
        pd_ = preds_at(float(tau), dev)
        ev = P.evaluate_predictions(pd_, refs, dev)
        score = (ev["objects"]["micro_f1"] + ev["clothing"]["micro_f1"]) / 2
        if score > best[0]:
            best = (score, float(tau))
    tau = best[1]
    # needs_review flags: below tau, or image-quality flags
    for k, v in obs.items():
        for o in v:
            reasons = []
            if o.category.value in CALIBRATED and o.uncertainty.combined_score is not None and o.uncertainty.combined_score < tau:
                reasons.append("low_score")
            if o.uncertainty.image_quality_flags:
                reasons.append("image_quality:" + "+".join(o.uncertainty.image_quality_flags))
            if o.canonical is None:
                reasons.append("unmapped_label")
            o.uncertainty.review_reasons = reasons
            o.uncertainty.needs_review = bool(reasons)

    # ---------------- methods
    methods: dict[str, dict] = {}
    methods["B5 structured VLM (no selection)"] = P.pred_sets({u: obs.get(u, []) for u in test})
    methods["B6 framework (struct + calibrated selection)"] = preds_at(tau, test)
    # high-precision operating point: smallest dev threshold whose retained OBJECT observations reach precision >= 0.95
    from vle.uncertainty.calibration import threshold_for_precision
    dev_obj = [r for r in dev_rows if r["category"] == "object"]
    tau_hp = threshold_for_precision(np.array([r["combined"] for r in dev_obj]), np.array([r["y"] for r in dev_obj]), 0.95)
    # applied to object observations only (the threshold is tuned on objects); other categories keep tau
    methods["B6-HP framework, high-precision mode"] = P.pred_sets(
        {k: obs.get(k, []) for k in test},
        keep=lambda o: (o.category.value not in CALIBRATED or o.uncertainty.combined_score is None
                        or o.uncertainty.combined_score >= (tau_hp if o.category.value == "object" else tau)))
    cap_run = RAW / "exp1" / "caption_b16"
    if (cap_run / "outputs.jsonl").exists():
        _, caprows = P.load_run(cap_run)
        cap = {r["meta"]["uid"]: P.caption_label_sets(r["text"]) for r in caprows}
        methods["B3 VLM free-form caption"] = {u: cap.get(u, {}) for u in test}
    bdir = RAW / "baselines" / "b1_b2"
    if (bdir / "blip_captions.jsonl").exists():
        blip = {json.loads(l)["uid"]: json.loads(l)["caption"] for l in (bdir / "blip_captions.jsonl").read_text().splitlines()}
        methods["B1 BLIP caption"] = {u: P.caption_label_sets(blip[u]) for u in test}
        cl = json.loads((bdir / "clip_scores.json").read_text())
        thr = tune_clip(cl["scores"], cl["label_space"], refs, dev)
        methods["B2 CLIP zero-shot"] = clip_preds(cl["scores"], cl["label_space"], thr, test)
    else:
        thr = None
    order = [m for m in ["B1 BLIP caption", "B2 CLIP zero-shot", "B3 VLM free-form caption",
                         "B5 structured VLM (no selection)", "B6 framework (struct + calibrated selection)",
                         "B6-HP framework, high-precision mode"] if m in methods]
    results = {m: full_eval(methods[m], refs, test) for m in order}

    # ---------------- paired tests framework vs others (per-image F1), Holm-corrected
    tests_out: dict[str, dict] = {}
    pvals = {}
    fw = results["B6 framework (struct + calibrated selection)"]["_per_image"]
    for m in order:
        if m.startswith("B6"):
            continue
        other = results[m]["_per_image"]
        for cat in ("objects", "clothing"):
            a, b = np.array(fw[cat]), np.array(other[cat])
            pb = S.paired_bootstrap(a, b)
            wx = S.wilcoxon(a, b)
            key = f"B6 vs {m.split()[0]} [{cat}]"
            tests_out[key] = {**pb, "wilcoxon_p": wx["p"], "cohens_dz": S.cohens_dz(a, b), "cliffs_delta": S.cliffs_delta(a, b),
                              "n": len(a)}
            pvals[key] = wx["p"]
    for k, v in S.holm(pvals).items():
        tests_out[k]["wilcoxon_p_holm"] = v

    # ---------------- Exp 6: calibration on test
    calib = {}
    signals = {"verbal (raw map)": "verbal_raw", "token prob (raw)": "token_raw", "verbal (Platt, dev)": "verbal_platt",
               "token (Platt, dev)": "token_platt", "CLIP (Platt, dev)": "ground_platt",
               "combined w/o CLIP": "combined_no_ground", "combined w/o quality": "combined_no_quality",
               "combined (LR, dev)": "combined"}
    for cat in ("object", "clothing", "all"):
        tr = [r for r in test_rows if cat == "all" or r["category"] == cat]
        if not tr:
            continue
        y = np.array([r["y"] for r in tr])
        calib[cat] = {"n_obs": len(tr), "n_images": len({r["uid"] for r in tr}), "base_precision": float(y.mean()), "signals": {}}
        for nm, key in signals.items():
            c = np.array([r[key] for r in tr])
            cov, risk, aurc = risk_coverage(c, y)
            calib[cat]["signals"][nm] = {
                "ece": ece(c, y), "ece_ci": cluster_bootstrap_metric(tr, key, ece),
                "brier": brier(c, y), "auroc": auroc(c, y),
                "auroc_ci": cluster_bootstrap_metric(tr, key, auroc),
                "aurc": aurc, "risk@50": selective_risk_at(c, y, 0.5), "risk@80": selective_risk_at(c, y, 0.8),
                "risk@100": float(1 - y.mean()), "reliability": reliability(c, y),
                "rc_curve": {"coverage": cov[:: max(1, len(cov) // 100)].tolist(), "risk": risk[:: max(1, len(cov) // 100)].tolist()}}
    calib["combiner_coefficients"] = variants["combined"].coefficients()

    # ---------------- Exp 7 (partial): structured-validation ablation
    statuses = defaultdict(int)
    naive = defaultdict(int)
    for u in uids:
        statuses[info.get(u, {}).get("parse_status", "missing")] += 1
        naive[info.get(u, {}).get("naive_status", "missing")] += 1
    # "without structured validation": images whose raw output is not plain-JSON parseable yield no observations
    novalid = {u: (obs.get(u, []) if info.get(u, {}).get("naive_status") == "valid" else []) for u in test}
    results_novalid = full_eval(P.pred_sets(novalid), refs, test)
    abl = {"parse_status_framework": dict(statuses), "parse_status_naive": dict(naive),
           "without_validation": {k: v for k, v in results_novalid.items() if not k.startswith("_")},
           "without_uncertainty": {k: v for k, v in results["B5 structured VLM (no selection)"].items() if not k.startswith("_")},
           "full": {k: v for k, v in results["B6 framework (struct + calibrated selection)"].items() if not k.startswith("_")}}
    # w/o quality/grounding signals in the selector
    for var in ("combined_no_ground", "combined_no_quality"):
        cvar = variants[var]
        sc = {}
        for k in dev + test:
            v = obs.get(k, [])
            sc[k] = cvar.predict([P.feature_row(o) for o in v]).tolist() if v else []
        best_v = (-1.0, 0.0)
        for t in np.arange(0.0, 0.9, 0.02):
            pdv = P.pred_sets({k: [o for o, s in zip(obs.get(k, []), sc[k]) if s >= t or o.category.value not in CALIBRATED] for k in dev})
            ev = P.evaluate_predictions(pdv, refs, dev)
            s_ = (ev["objects"]["micro_f1"] + ev["clothing"]["micro_f1"]) / 2
            if s_ > best_v[0]:
                best_v = (s_, float(t))
        pt = P.pred_sets({k: [o for o, s in zip(obs.get(k, []), sc[k]) if s >= best_v[1] or o.category.value not in CALIBRATED] for k in test})
        r_ = full_eval(pt, refs, test)
        abl[f"selector_{var}"] = {"tau": best_v[1], **{k: v for k, v in r_.items() if not k.startswith("_")}}
        abl[f"selector_{var}"]["_per_image"] = r_["_per_image"]

    # per-image vectors for paired ablation tests
    full_pi = results["B6 framework (struct + calibrated selection)"]["_per_image"]
    abl_tests = {}
    for nm, other in [("w/o uncertainty selection", results["B5 structured VLM (no selection)"]["_per_image"]),
                      ("w/o structured validation", results_novalid["_per_image"]),
                      ("w/o CLIP grounding signal", abl["selector_combined_no_ground"]["_per_image"]),
                      ("w/o image-quality signal", abl["selector_combined_no_quality"]["_per_image"])]:
        for cat in ("objects", "clothing"):
            a, b = np.array(full_pi[cat]), np.array(other[cat])
            abl_tests[f"{nm} [{cat}]"] = {**S.paired_bootstrap(a, b), "wilcoxon_p": S.wilcoxon(a, b)["p"],
                                          "cohens_dz": S.cohens_dz(a, b)}
    adj = S.holm({k: v["wilcoxon_p"] for k, v in abl_tests.items()})
    for k in abl_tests:
        abl_tests[k]["wilcoxon_p_holm"] = adj[k]
    for k in ("selector_combined_no_ground", "selector_combined_no_quality"):
        abl[k].pop("_per_image", None)
    abl["paired_tests"] = abl_tests

    # ---------------- review-flag statistics (input for Exp 8 simulation)
    flag_rate = np.mean([o.uncertainty.needs_review for u in test for o in obs.get(u, [])]) if test else None

    # ---------------- latency/throughput of the full 1,000-image run
    st = sorted(run.glob("stats_*.json"))
    run_stats = json.loads(st[-1].read_text()) if st else None

    summary = {
        "n_images": {"all": len(uids), "dev": len(dev), "test": len(test), "missing_outputs": missing},
        "selection_tau_dev": tau, "selection_tau_hp_dev": float(tau_hp), "clip_thresholds_dev": thr,
        "methods": {m: {k: v for k, v in results[m].items() if not k.startswith("_")} for m in order},
        "paired_tests": tests_out, "review_flag_rate_test": flag_rate, "run_stats": {k: v for k, v in (run_stats or {}).items() if k not in ("batch_sizes", "batch_latencies")},
        "output_tokens_mean": float(np.mean([i["n_output_tokens"] or 0 for i in info.values()])),
    }
    PROC.mkdir(parents=True, exist_ok=True)
    write_json(PROC / "exp1_summary.json", summary)
    write_json(PROC / "exp6_summary.json", calib)
    write_json(PROC / "exp7_partial_summary.json", abl)
    write_json(PROC / "exp1_per_image.json", {m: results[m]["_per_image"] for m in order})
    # observations with calibrated scores for the review store and Exp 3/8
    with open(PROC / "exp1_observations.jsonl", "w", encoding="utf-8") as f:
        for u in uids:
            for o in obs.get(u, []):
                f.write(o.model_dump_json() + "\n")
    write_json(PROC / "exp1_image_info.json", {u: {**{k: v for k, v in info[u].items() if k != "text"},
                                                    "quality": quality[u].model_dump()} for u in info})
    # combiner persisted for downstream experiments
    import pickle
    (PROC / "combiner.pkl").write_bytes(pickle.dumps({"combined": variants["combined"], "tau": tau}))

    # ---------------- tables
    trows = []
    for m in order:
        r = results[m]
        trows.append([esc(m), ci(r["objects"]["p"], *r["objects"]["p_ci"]), ci(r["objects"]["r"], *r["objects"]["r_ci"]),
                      ci(r["objects"]["f1"], *r["objects"]["f1_ci"]), fmt(r["objects_macro_f1"]),
                      ci(r["chair"]["chair_i"], *r["chair"]["chair_i_ci"])])
    write_table("exp1_objects", ["Method", "Precision", "Recall", "Micro-F1", "Macro-F1", r"CHAIR$_i$"], trows,
                "Object extraction on the test split (COCO and VQA~v2 images, exhaustive COCO-80 references; "
                f"$n={results[order[0]]['objects']['n_images']}$ images). 95\\% bootstrap CIs in brackets.", "tab:exp1_objects",
                colspec="lccccc", size=r"\scriptsize")
    trows = []
    for m in order:
        r = results[m]
        trows.append([esc(m), ci(r["clothing"]["p"], *r["clothing"]["p_ci"]), ci(r["clothing"]["r"], *r["clothing"]["r_ci"]),
                      ci(r["clothing"]["f1"], *r["clothing"]["f1_ci"]), fmt(r["color"]["color_acc"]) + f" ({r['color']['n_pairs']})",
                      fmt(r["accessory"]["f1"]), fmt(r["scene"]["hit_rate"])])
    write_table("exp1_attributes", ["Method", "Cloth. P", "Cloth. R", "Cloth. F1", "Colour acc. (n)", "Access. F1", "Scene hit"], trows,
                "Clothing, accessory and scene extraction on the test split. Clothing/accessory references come from Visual Genome "
                f"and Flickr30K Entities ($n={results[order[0]]['clothing']['n_images']}$ images) and are not exhaustive, so precision is a lower bound. "
                "Scene hit rate: predicted place among places named by $\\geq2$ human captions.", "tab:exp1_attributes",
                colspec="lcccccc", size=r"\scriptsize")
    crow = []
    for nm, s in calib["object"]["signals"].items():
        crow.append([esc(nm), ci(s["ece"], *s["ece_ci"]), fmt(s["brier"]), ci(s["auroc"], *s["auroc_ci"]), fmt(s["aurc"]), fmt(s["risk@80"])])
    write_table("exp6_calibration", ["Confidence signal", "ECE", "Brier", "AUROC", "AURC", "Risk@80\\%"], crow,
                f"Calibration and selective prediction for object observations on the test split "
                f"($n={calib['object']['n_obs']}$ observations, {calib['object']['n_images']} images; base precision "
                f"{calib['object']['base_precision']:.3f}). Calibrators are fit on the dev split; CIs by image-level cluster bootstrap.",
                "tab:exp6", colspec="lccccc", size=r"\scriptsize")
    # ---------------- main comparison table (all methods, all categories) with significance vs B6
    def sig(m: str, cat: str) -> str:
        k = f"B6 vs {m.split()[0]} [{cat}]"
        if k not in tests_out:
            return ""
        p = tests_out[k]["wilcoxon_p_holm"]
        return "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""
    short = {"B1 BLIP caption": "B1 BLIP-base caption", "B2 CLIP zero-shot": "B2 CLIP ViT-B/32 zero-shot",
             "B3 VLM free-form caption": "B3 Qwen2.5-VL caption", "B5 structured VLM (no selection)": "B5 Qwen2.5-VL structured",
             "B6 framework (struct + calibrated selection)": "B6 Framework (ours)",
             "B6-HP framework, high-precision mode": "B6-HP Framework, high-precision"}
    rows_c = []
    for m in order:
        r = results[m]
        col = r["color"]["color_acc"]
        rows_c.append([esc(short.get(m, m)), fmt(r["objects"]["p"]), fmt(r["objects"]["r"]),
                       fmt(r["objects"]["f1"]) + sig(m, "objects"), fmt(r["objects_macro_f1"]),
                       fmt(r["chair"]["chair_i"]), fmt(r["clothing"]["p"]), fmt(r["clothing"]["r"]),
                       fmt(r["clothing"]["f1"]) + sig(m, "clothing"),
                       "--" if col != col else fmt(col), fmt(r["accessory"]["f1"]), fmt(r["carried"]["f1"]),
                       "--" if m.startswith("B2") else fmt(r["scene"]["hit_rate"])])
    write_table("exp1_comparison",
                [r"\multirow{2}{*}{Method} & \multicolumn{5}{c}{Objects (COCO/VQA)} & \multicolumn{4}{c}{Garments (VG/Flickr30K)} & Access. & Carried & Scene \\ \cmidrule(lr){2-6}\cmidrule(lr){7-10}", "P", "R", "F1", "macro-F1", r"CHAIR$_i\downarrow$",
                 "P", "R", "F1", "colour", "F1", "F1", "hit"],
                rows_c,
                "Comparison with baselines on the test split (E1). Objects: COCO/VQA images with exhaustive COCO-80 references "
                f"($n={results[order[0]]['objects']['n_images']}$). Garments, accessories, carried objects: VG/Flickr30K images "
                f"($n={results[order[0]]['clothing']['n_images']}$; references not exhaustive, so precision is a lower bound). "
                "Scene hit: predicted place among places named by $\\geq2$ captions. Colour accuracy only for methods that output "
                "garment colours. 95\\% bootstrap CIs are given in the text and in Online Resource~1. Per-image F1 compared with B6 by Wilcoxon signed-rank test, Holm-adjusted: $^{*}p<0.05$, $^{**}p<0.01$.",
                "tab:comparison", colspec="lcccccccccccc")
    # ---------------- ablation table (E7)
    abl_rows = []
    for nm, key, tkey in (("Full framework (B6)", "full", None), ("w/o uncertainty-based selection", "without_uncertainty", "w/o uncertainty selection"),
                          ("w/o structured validation (no JSON repair)", "without_validation", "w/o structured validation"),
                          ("w/o CLIP signal in calibrator", "selector_combined_no_ground", "w/o CLIP grounding signal"),
                          ("w/o image-quality signal in calibrator", "selector_combined_no_quality", "w/o image-quality signal")):
        a_ = abl[key]
        def dcell(cat: str) -> str:
            if tkey is None:
                return "--"
            t_ = abl["paired_tests"][f"{tkey} [{cat}]"]
            return f"{t_['mean_diff']:+.3f} (p={t_['wilcoxon_p_holm']:.2g})"
        abl_rows.append([esc(nm), fmt(a_["objects"]["f1"]), fmt(a_["chair"]["chair_i"]), dcell("objects"),
                         fmt(a_["clothing"]["f1"]), dcell("clothing")])
    write_table("exp7_ablation", ["Configuration", "Obj. F1", r"CHAIR$_i$", r"$\Delta$F1 full$-$variant (obj.)",
                                  "Garm. F1", r"$\Delta$F1 full$-$variant (garm.)"], abl_rows,
                "Ablation of framework components on the test split (E7). $\\Delta$F1: mean per-image difference (full minus "
                "ablated), Wilcoxon signed-rank $p$ Holm-adjusted over the eight comparisons. Without validation, "
                f"{naive.get('failed', 0)} of 1,000 raw outputs are unusable, versus {statuses.get('failed', 0)} after repair. "
                "Scheduler and aggregation ablations are reported in Fig.~\\ref{fig:tput} and Table~\\ref{tab:exp3}.",
                "tab:ablation", colspec="lccccc")
    print(json.dumps({m: (results[m]["objects"]["f1"], results[m]["clothing"]["f1"], results[m]["chair"]["chair_i"]) for m in order}, indent=1))
    print("tau", tau, "parse", dict(statuses), "naive", dict(naive))


if __name__ == "__main__":
    main()
