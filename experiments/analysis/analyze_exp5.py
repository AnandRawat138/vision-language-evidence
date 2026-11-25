"""Experiment 5: robustness to controlled image degradation (200 COCO/VQA test images).

Conditions: original (from exp1/struct_b16) and blur, downscale, lowlight, occlusion, jpeg at two severities
(results/raw/exp5/<kind>_<level>/). Reports object F1/recall/CHAIR, paired change vs original, quality-flag
firing rates, model self-reported quality, and precision retained by calibrated selection.

    python experiments/analysis/analyze_exp5.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from run_inference import subset_rows  # noqa: E402
from vle.evaluation import pipeline as P  # noqa: E402
from vle.evaluation.metrics import prf  # noqa: E402
from vle.preprocessing.quality import assess_quality, degrade, load_rgb  # noqa: E402
from vle.reporting.tables import ci, esc, fmt, write_table  # noqa: E402
from vle.runio import RAW, write_json  # noqa: E402
from vle.stats import tests as S  # noqa: E402

PROC = ROOT / "results" / "processed"
CONDS = ["blur:2", "blur:5", "downscale:0.25", "downscale:0.1", "lowlight:2.0", "lowlight:3.5",
         "occlusion:0.25", "occlusion:0.5", "jpeg:20", "jpeg:5"]


def run_name(c: str) -> str:
    return c.replace(":", "_")


def main() -> None:
    man = P.manifest_index()
    uids = [r["uid"] for r in subset_rows("robust")]
    refs = P.references(uids)
    pk = pickle.loads((PROC / "combiner.pkl").read_bytes())
    comb, tau = pk["combined"], pk["tau"]

    def get_obs(run: Path, cond: str | None):  # type: ignore[no-untyped-def]
        obs, info = P.run_observations(run, None)
        obs = {u: obs.get(u, []) for u in uids}
        imgs, flags = {}, {}
        for u in uids:
            base = load_rgb(ROOT / man[u]["local_path"])
            if cond:
                k, lv = cond.split(":")
                base = degrade(base, k, float(lv), seed=int(man[u]["md5"][:8], 16) % 10_000)
            imgs[u] = base
            flags[u] = assess_quality(base).flags
        for u, v in obs.items():
            for o in v:
                o.uncertainty.image_quality_flags = flags[u]
        P.add_grounding(obs, imgs, f"exp5_{run_name(cond) if cond else 'orig'}")
        for u, v in obs.items():
            if v:
                for o, s in zip(v, comb.predict([P.feature_row(o) for o in v])):
                    o.uncertainty.combined_score = float(s)
        return obs, info, flags

    def metrics(obs: dict) -> dict:
        f1s, rec, hal, sel_prec_n, sel_prec_d, n_kept, n_all = [], [], [], 0, 0, 0, 0
        for u in uids:
            g = set(refs[u]["objects"])
            pr = {o.canonical for o in obs[u] if o.category.value == "object" and o.canonical}
            ks = {o.canonical for o in obs[u] if o.category.value == "object" and o.canonical and o.uncertainty.combined_score >= tau}
            f1s.append(1.0 if not pr and not g else prf(len(pr & g), len(pr - g), len(g - pr))[2])
            rec.append(len(pr & g) / len(g) if g else 1.0)
            hal.append(len(pr - g) / len(pr) if pr else 0.0)
            sel_prec_n += len(ks & g)
            sel_prec_d += len(ks)
            n_kept += len(ks)
            n_all += len(pr)
        allp = sum(len({o.canonical for o in obs[u] if o.category.value == "object" and o.canonical} & set(refs[u]["objects"])) for u in uids)
        return {"f1": np.array(f1s), "recall": np.array(rec), "hall": np.array(hal),
                "precision_all": allp / max(n_all, 1), "precision_selected": sel_prec_n / max(sel_prec_d, 1),
                "coverage_selected": n_kept / max(n_all, 1)}

    base_obs, base_info, base_flags = get_obs(RAW / "exp1" / "struct_b16", None)
    base = metrics(base_obs)
    out = {"n_images": len(uids), "tau": tau, "conditions": {}}

    def summarize(name, mm, flags, info):  # type: ignore[no-untyped-def]
        vq = [info.get(u, {}).get("vlm_quality") or {} for u in uids]
        return {"f1": S.describe(mm["f1"]), "recall": S.describe(mm["recall"]), "hall": S.describe(mm["hall"]),
                "precision_all": mm["precision_all"], "precision_selected": mm["precision_selected"],
                "coverage_selected": mm["coverage_selected"],
                "quality_flag_rate": float(np.mean([bool(flags[u]) for u in uids])),
                "vlm_reports_degradation": float(np.mean([(q.get("blur") not in (None, "none")) or (q.get("lighting") not in (None, "good"))
                                                          or (q.get("occlusion") not in (None, "none")) for q in vq])),
                "parse_failed": int(sum(info.get(u, {}).get("parse_status") == "failed" for u in uids))}

    out["conditions"]["original"] = summarize("original", base, base_flags, base_info)
    pvals = {}
    for c in CONDS:
        run = RAW / "exp5" / run_name(c)
        if not (run / "outputs.jsonl").exists():
            out["conditions"][c] = {"status": "not run"}
            continue
        o, info, fl = get_obs(run, c)
        mm = metrics(o)
        d = summarize(c, mm, fl, info)
        d["delta_f1_vs_original"] = S.paired_bootstrap(mm["f1"], base["f1"])
        d["delta_f1_vs_original"]["wilcoxon_p"] = S.wilcoxon(mm["f1"], base["f1"])["p"]
        d["relative_f1_change"] = (mm["f1"].mean() - base["f1"].mean()) / base["f1"].mean()
        pvals[c] = d["delta_f1_vs_original"]["wilcoxon_p"]
        out["conditions"][c] = d
    for c, p in S.holm(pvals).items():
        out["conditions"][c]["delta_f1_vs_original"]["wilcoxon_p_holm"] = p
    write_json(PROC / "exp5_summary.json", out)
    rows = []
    for c, d in out["conditions"].items():
        if "f1" not in d:
            continue
        dd = d.get("delta_f1_vs_original")
        rows.append([esc(c), ci(d["f1"]["mean"], d["f1"]["ci95_lo"], d["f1"]["ci95_hi"]),
                     "--" if not dd else f"{dd['mean_diff']:+.3f} [{dd['ci95_lo']:+.3f}, {dd['ci95_hi']:+.3f}]",
                     fmt(d["hall"]["mean"]), fmt(d["precision_all"]), fmt(d["precision_selected"]), fmt(d["coverage_selected"]),
                     fmt(d["quality_flag_rate"], 2)])
    write_table("exp5_robustness", ["Condition", "Object F1", r"$\Delta$F1 vs orig.", "Halluc.", "Prec. (all)", "Prec. (sel.)",
                                    "Coverage", "QA flag"], rows,
                f"Robustness to controlled degradations ({len(uids)} COCO/VQA test images). Prec.~(sel.) is precision after "
                "calibrated selective reporting with the dev-selected threshold; Coverage is the retained fraction of object "
                "observations; QA flag is the fraction of images flagged by the image-quality module.", "tab:exp5",
                colspec="lccccccc", size=r"\scriptsize")
    print(json.dumps({c: (d.get("f1", {}).get("mean"), d.get("precision_selected")) for c, d in out["conditions"].items()}, indent=1))


if __name__ == "__main__":
    main()
