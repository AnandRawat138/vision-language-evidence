"""Experiment 8 (partial): ORACLE-SIMULATED review-prioritisation analysis.

No human participants were involved. This script does NOT produce human-evaluation results.
It uses the reference annotations as an oracle "reviewer" to quantify how much review effort
an uncertainty-ranked review queue would save compared with reviewing observations in random
order, i.e. it evaluates the ranking that the review interface presents to humans (H3, partly).

Metrics (test split, object + clothing observations with references):
  * errors found after reviewing k% of observations: uncertainty-ranked vs random (mean of 200 shuffles)
  * area under the error-discovery curve (normalised)
  * observations needing review flagged by the framework and their error precision

The real human-in-the-loop study protocol is documented in docs/human_evaluation_protocol.md.

    python experiments/human_evaluation/simulate_review.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from vle.evaluation import pipeline as P  # noqa: E402
from vle.reporting.tables import fmt, write_table  # noqa: E402
from vle.runio import write_json  # noqa: E402
from vle.schemas import Observation  # noqa: E402

PROC = ROOT / "results" / "processed"


def main() -> None:
    man = P.manifest_index()
    obs: dict[str, list[Observation]] = {}
    for line in (PROC / "exp1_observations.jsonl").read_text(encoding="utf-8").splitlines():
        o = Observation.model_validate_json(line)
        obs.setdefault(o.provenance.image_id, []).append(o)
    test = [u for u in obs if man[u]["split"] == "test"]
    refs = P.references(test)
    rows = P.labelled_rows({u: obs[u] for u in test}, refs)
    y_err = np.array([1 - r["y"] for r in rows])
    score = np.array([r["obs"].uncertainty.combined_score for r in rows])
    flagged = np.array([r["obs"].uncertainty.needs_review for r in rows])
    n, E = len(rows), y_err.sum()
    order = np.argsort(score)  # least confident first
    found_rank = np.cumsum(y_err[order]) / E
    rng = np.random.default_rng(2026)
    rand = np.mean([np.cumsum(y_err[rng.permutation(n)]) / E for _ in range(200)], axis=0)
    ks = [0.1, 0.2, 0.3, 0.5]
    res = {"simulation": "oracle reviewer using dataset reference annotations; NOT a human study",
           "n_observations": int(n), "n_errors": int(E), "error_rate": float(E / n),
           "errors_found_ranked": {f"{int(k * 100)}%": float(found_rank[int(k * n) - 1]) for k in ks},
           "errors_found_random": {f"{int(k * 100)}%": float(rand[int(k * n) - 1]) for k in ks},
           "auc_ranked": float(found_rank.mean()), "auc_random": float(rand.mean()),
           "flag_rate": float(flagged.mean()),
           "flag_precision_errors": float(y_err[flagged].mean()) if flagged.any() else None,
           "flag_recall_errors": float(y_err[flagged].sum() / E) if E else None,
           "curve": {"x": (np.arange(1, n + 1) / n)[:: max(1, n // 200)].tolist(),
                     "ranked": found_rank[:: max(1, n // 200)].tolist(), "random": rand[:: max(1, n // 200)].tolist()}}
    # within-category analysis (the pooled ranking partly separates categories with different error rates)
    res["per_category"] = {}
    for cat in ("object", "clothing"):
        idx = np.array([r["category"] == cat for r in rows])
        ye, sc = y_err[idx], score[idx]
        if ye.sum() == 0:
            continue
        fr = np.cumsum(ye[np.argsort(sc)]) / ye.sum()
        rnd = np.mean([np.cumsum(ye[rng.permutation(len(ye))]) / ye.sum() for _ in range(200)], axis=0)
        k30 = int(0.3 * len(ye)) - 1
        res["per_category"][cat] = {"n": int(len(ye)), "errors": int(ye.sum()), "auc_ranked": float(fr.mean()),
                                    "auc_random": float(rnd.mean()), "found_at_30%_ranked": float(fr[k30]),
                                    "found_at_30%_random": float(rnd[k30])}
    res["human_study"] = {"status": "incomplete", "reason": "no authorised reviewers or ethics approval were available; "
                          "reviewer correction rate, review time and inter-annotator agreement were not measured"}
    write_json(PROC / "exp8_summary.json", res)
    write_table("exp8_review", ["Reviewed fraction", "Errors found (uncertainty-ranked)", "Errors found (random order)"],
                [[k.replace("%", r"\%"), fmt(res["errors_found_ranked"][k]), fmt(res["errors_found_random"][k])] for k in res["errors_found_ranked"]],
                f"Oracle-simulated review prioritisation on test observations ($n={n}$, {int(E)} reference-inconsistent). "
                "This is a simulation using dataset annotations, not a human study.", "tab:exp8", colspec="lcc")
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
