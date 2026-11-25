"""Qualitative test cases (Fig. 6): real TEST-split images with framework outputs versus references.

Only images whose licence permits reuse with attribution are eligible: COCO images under
"Attribution License" (CC BY 2.0) and Visual Genome images (CC BY 4.0). Cases are selected
deterministically by rule (not by hand-picking favourable outputs):
  case A: COCO CC-BY image of the Exp.-5 subset with >=3 reference objects and no false positives
  case B: COCO CC-BY image with >=1 false positive object (typical failure)
  case C: VG image with >=2 reference garments (clothing/colour behaviour)
  case D: case-A image under Gaussian blur sigma=5 (Exp. 5 output), showing degradation and flags
Within each rule the first image in uid order is taken. The same content is written to
docs/test_cases.md and results/processed/test_cases.json.

    python scripts/make_case_figure.py
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vle.datasets.manifest import load_manifest  # noqa: E402
from vle.datasets.reference import load_reference  # noqa: E402
from vle.evaluation import pipeline as P  # noqa: E402
from vle.preprocessing.quality import assess_quality, degrade, load_rgb  # noqa: E402
from vle.runio import RAW  # noqa: E402
from vle.schemas import Observation  # noqa: E402

PROC = ROOT / "results" / "processed"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 8, "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
COL = {"TP": "#2f7d46", "FP": "#b83b3b", "FN": "#6b6b6b", "n/a": "#2b5c8a"}


def load_obs() -> dict[str, list[Observation]]:
    out: dict[str, list[Observation]] = {}
    for line in (PROC / "exp1_observations.jsonl").read_text(encoding="utf-8").splitlines():
        o = Observation.model_validate_json(line)
        out.setdefault(o.provenance.image_id, []).append(o)
    return out


def rows_for(obs: list[Observation], ref: dict) -> list[tuple[str, str]]:
    """(tag, text) lines: objects/garments with calibrated score and correctness vs reference."""
    lines: list[tuple[str, str]] = []
    seen = set()
    for o in obs:
        if o.category.value not in ("object", "clothing") or o.canonical is None:
            continue
        key = (o.category.value, o.canonical, o.attribute)
        if key in seen:
            continue
        seen.add(key)
        if o.category.value == "object":
            tag = ("TP" if o.canonical in ref["objects"] else "FP") if ref["objects_complete"] else "n/a"
            name = o.canonical
        else:
            tag = ("TP" if o.canonical in ref["clothing"] else "FP") if ref.get("attr_available") else "n/a"
            name = f"{o.attribute + ' ' if o.attribute else ''}{o.canonical}"
        p = o.uncertainty.combined_score
        flag = " [review]" if o.uncertainty.needs_review else ""
        lines.append((tag, f"{tag:>3}  {name}  p={p:.2f}{flag}" if p is not None else f"{tag:>3}  {name}{flag}"))
    pred_o = {o.canonical for o in obs if o.category.value == "object"}
    if ref["objects_complete"]:
        for m in sorted(set(ref["objects"]) - pred_o):
            lines.append(("FN", f" FN  {m}  (missed)"))
    if ref.get("attr_available"):
        pred_c = {o.canonical for o in obs if o.category.value == "clothing"}
        for g, cols in ref["clothing"].items():
            if g not in pred_c:
                lines.append(("FN", f" FN  {'/'.join(cols) + ' ' if cols else ''}{g}  (missed)"))
    return lines


def main() -> None:
    man = {r["uid"]: r for r in load_manifest(ROOT)}
    obs = load_obs()
    test = sorted(u for u in man if man[u]["split"] == "test" and u in obs)
    refs = {u: load_reference(ROOT, u) for u in test}

    def fp_count(u: str) -> int:
        return len({o.canonical for o in obs[u] if o.category.value == "object" and o.canonical} - set(refs[u]["objects"]))

    # case A is drawn from the Exp.-5 robustness subset so that its degraded version (case D) exists
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_inference import subset_rows
    robust = {r["uid"] for r in subset_rows("robust")}
    cc_coco = [u for u in test if man[u]["source"] == "coco" and man[u]["license"] == "Attribution License"]
    cc_robust = [u for u in cc_coco if u in robust]
    a = next(u for u in cc_robust if len(refs[u]["objects"]) >= 3 and fp_count(u) == 0)
    b = next(u for u in cc_coco if fp_count(u) >= 1 and u != a)
    c = next(u for u in test if man[u]["source"] == "vg" and len(refs[u]["clothing"]) >= 2)
    cases = [("A", a, None), ("B", b, None), ("C", c, None)]

    # case D: blurred version of A from the Exp. 5 run (if available)
    blur_dir = RAW / "exp5" / "blur_5"
    if (blur_dir / "outputs.jsonl").exists():
        bobs, _ = P.run_observations(blur_dir, None)
        if a in bobs:
            import pickle
            comb = pickle.loads((PROC / "combiner.pkl").read_bytes())
            img = degrade(load_rgb(ROOT / man[a]["local_path"]), "blur", 5.0, seed=int(man[a]["md5"][:8], 16) % 10_000)
            flags = assess_quality(img).flags
            for o in bobs[a]:
                o.uncertainty.image_quality_flags = flags
            P.add_grounding({a: bobs[a]}, {a: img}, "exp5_blur_5")
            if bobs[a]:
                for o, s in zip(bobs[a], comb["combined"].predict([P.feature_row(o) for o in bobs[a]])):
                    o.uncertainty.combined_score = float(s)
                    o.uncertainty.needs_review = bool(s < comb["tau"] or flags)
            obs["__blur__"] = bobs[a]
            cases.append(("D", a, "blur:5"))

    n = len(cases)
    fig, axs = plt.subplots(2, n, figsize=(6.85, 3.7), gridspec_kw={"height_ratios": [1.45, 1.0]})
    record = []
    for i, (letter, u, deg) in enumerate(cases):
        img = load_rgb(ROOT / man[u]["local_path"])
        if deg:
            k, lv = deg.split(":")
            img = degrade(img, k, float(lv), seed=int(man[u]["md5"][:8], 16) % 10_000)
        q = assess_quality(img)
        o_list = obs["__blur__"] if deg else obs[u]
        lines = rows_for(o_list, refs[u])
        ax = axs[0, i]
        # letterbox into an identical 4:3 frame so all panels have the same size and aligned titles
        from PIL import Image as _Im
        frame = _Im.new("RGB", (800, 600), "white")
        im2 = img.copy()
        im2.thumbnail((800, 600))
        frame.paste(im2, ((800 - im2.size[0]) // 2, (600 - im2.size[1]) // 2))
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        src = man[u]["source"].upper() if man[u]["source"] != "vg" else "VG"
        ax.set_title(f"({letter}) {src} {man[u]['source_id']}" + (f", {deg.replace(':', ' σ=')}" if deg else ""), fontsize=8)
        tx = axs[1, i]
        tx.axis("off")
        shown = lines[:11]
        y = 1.0
        for tag, t in shown:
            tx.text(0.0, y, textwrap.shorten(t, 34, placeholder="…"), color=COL[tag], fontsize=7, family="serif",
                    va="top", ha="left", transform=tx.transAxes)
            y -= 0.1
        if len(lines) > len(shown):
            tx.text(0.0, y, f"… {len(lines) - len(shown)} more lines in test_cases.md", fontsize=6.5, va="top",
                    color="#444444", transform=tx.transAxes)
            y -= 0.1
        tx.text(0.0, y - 0.02, f"quality flags: {', '.join(q.flags) or 'none'}", fontsize=6.5, va="top", style="italic",
                transform=tx.transAxes)
        record.append({"case": letter, "uid": u, "source": man[u]["source"], "source_id": man[u]["source_id"],
                       "licence": man[u]["license"], "url": man[u]["url"], "degradation": deg, "quality_flags": q.flags,
                       "lines": [t for _, t in lines], "reference_objects": refs[u]["objects"],
                       "reference_clothing": refs[u]["clothing"]})
    fig.tight_layout(h_pad=0.4, w_pad=0.8)
    for d in (ROOT / "results" / "figures", ROOT / "paper" / "figures"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "fig6_test_cases.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "results" / "figures" / "fig6_test_cases.png", dpi=600, bbox_inches="tight")
    (PROC / "test_cases.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    md = ["# Qualitative test cases (test split)", "",
          "Selection rules are in `scripts/make_case_figure.py`. TP = label in the original reference annotation; "
          "FP = not in an exhaustive reference (hallucination); FN = reference label not produced; n/a = no exhaustive reference. "
          "p = calibrated score; [review] = flagged for human review.", ""]
    for r in record:
        md += [f"## Case {r['case']}: {r['uid']}" + (f" ({r['degradation']})" if r["degradation"] else ""),
               f"* Source: {r['source']} id {r['source_id']}, licence: {r['licence']}, URL: {r['url']}",
               f"* Quality flags: {', '.join(r['quality_flags']) or 'none'}",
               f"* Reference objects: {', '.join(r['reference_objects']) or '—'}; reference garments: {r['reference_clothing'] or '—'}",
               "", "```", *r["lines"], "```", ""]
    (ROOT / "docs" / "test_cases.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("case", "uid", "licence")} for r in record], indent=1))


if __name__ == "__main__":
    main()
