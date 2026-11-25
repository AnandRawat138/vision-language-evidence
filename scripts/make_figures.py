"""Generate all manuscript figures (PDF vector + 600-dpi PNG, Times New Roman, white background).

Data figures read only results/processed/*.json; diagrams are drawn from the implemented module structure.

    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROC = ROOT / "results" / "processed"
OUT = ROOT / "results" / "figures"
PAPER = ROOT / "paper" / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "pdf.fonttype": 42})
# Okabe-Ito colour-blind-safe palette
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "gray": "#5A5A5A",
     "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442",
     "light": "#e8eef5", "lgreen": "#e6f2ea", "lorange": "#fbeedf", "lred": "#f6e3e3", "lgray": "#efefef"}


def save(fig, name: str) -> None:  # type: ignore[no-untyped-def]
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    for d in (OUT, PAPER):
        fig.savefig(d / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig)
    print("saved", name)


def load(name: str) -> dict | None:
    p = PROC / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------- diagrams ---
def diagrams() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import diagrams as D
    for name, fn in (("fig1_architecture", D.fig_architecture), ("fig2_narrative_workflow", D.fig_narrative),
                     ("fig3_batch_architecture", D.fig_batching), ("fig4_aggregation", D.fig_aggregation),
                     ("fig5_uncertainty_workflow", D.fig_uncertainty)):
        save(fn(), name)

# ---------------------------------------------------------------- data ----
# Rules: legends are always placed OUTSIDE the plotting area (above the axes) so they cannot hide data;
# series differ by colour AND marker/line style/hatch (greyscale-safe); panels are labelled (a), (b), ...;
# no titles inside figures (captions are in the manuscript).
MARK = ["o", "s", "^", "D", "v", "P"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
HATCH = ["", "///", "...", "xx", "\\\\", "--"]


def legend_above(ax, ncol: int = 2, fs: float = 7) -> None:  # type: ignore[no-untyped-def]
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=ncol, frameon=False, fontsize=fs,
              handlelength=2.2, columnspacing=1.0, borderaxespad=0.2)


def panel(ax, letter: str) -> None:  # type: ignore[no-untyped-def]
    ax.text(-0.16, 1.0, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def log2_axis(ax, B) -> None:  # type: ignore[no-untyped-def]
    ax.set_xscale("log", base=2)
    ax.set_xticks(B, [str(b) for b in B])
    ax.minorticks_off()
    ax.set_xlabel("Batch size B")


def fig_throughput_latency() -> None:
    d = load("exp2_summary")
    if not d:
        return
    rows = [r for r in d["rows"] if r["policy"] in ("independent", "static")]
    B = [r["batch"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(6.85, 2.6))
    ax = axs[0]
    ax.errorbar(B, [r["throughput_mean"] for r in rows], yerr=[r["throughput_sd"] for r in rows], marker="o", ms=4,
                color=C["blue"], capsize=2, lw=1, label="static / independent")
    for r in d["rows"]:
        if r["policy"] == "adaptive":
            ax.axhline(r["throughput_mean"], color=C["green"], ls="--", lw=0.9, label="adaptive (incl. ramp-up)")
        if r["policy"] == "sorted":
            ax.plot([16], [r["throughput_mean"]], marker="s", ms=5, ls="none", color=C["orange"], label="length-sorted, B=16")
    log2_axis(ax, B)
    ax.set_ylabel("Throughput (images/s)")
    ax.set_ylim(bottom=0)
    legend_above(ax, 1)
    panel(ax, "a")
    ax = axs[1]
    # every request in a batch waits for the whole batch, so p50/p95/p99 coincide within a configuration;
    # plot the median service latency and the GPU time per image instead (log scale)
    ax.plot(B, [r["latency_p50"] for r in rows], marker="o", ms=4, color=C["blue"], lw=1, label="service latency (median)")
    ax.plot(B, [r["latency_p50"] / r["batch"] for r in rows], marker="s", ms=4, ls="--", color=C["red"], lw=1,
            label="GPU time per image")
    log2_axis(ax, B)
    ax.set_yscale("log")
    ax.set_ylabel("Seconds (log scale)")
    legend_above(ax, 1)
    panel(ax, "b")
    ax = axs[2]
    ax.plot(B, [r["torch_peak_alloc_mb"] / 1024 for r in rows], marker="o", ms=4, color=C["blue"], lw=1, label="PyTorch peak")
    ax.plot(B, [r["gpu_mem_peak_mb"] / 1024 for r in rows], marker="^", ms=4, ls="--", color=C["gray"], lw=1, label="device (NVML)")
    ax.axhline(6.0, color=C["red"], ls=":", lw=0.9, label="6 GB capacity")
    if B[-1] == 64:
        ax.annotate("spills to shared system memory", xy=(58, rows[-1]["torch_peak_alloc_mb"] / 1024 + 0.05),
                    xytext=(1.2, 6.75), fontsize=6.5, va="center", color=C["gray"],
                    arrowprops={"arrowstyle": "->", "lw": 0.7, "color": C["gray"]})
    log2_axis(ax, B)
    ax.set_ylabel("GPU memory (GB)")
    ax.set_ylim(0, 7.2)
    legend_above(ax, 2, 6.5)
    panel(ax, "c")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7_8_throughput_latency_memory")
    on = d.get("online", {}).get("rows")
    if on:
        fig, ax = plt.subplots(figsize=(3.31, 2.6))
        labels = [r["config"].replace("online_", "").replace("indep_", "independent\n").replace("dyn8_", "dynamic B≤8\n")
                  .replace("Hz", " /s") for r in on]
        x = np.arange(len(on))
        ax.bar(x - 0.2, [r["latency_p50"] for r in on], 0.4, color=C["blue"], label="p50", edgecolor="black", lw=0.4)
        ax.bar(x + 0.2, [r["latency_p95"] for r in on], 0.4, color=C["orange"], hatch="///", label="p95", edgecolor="black", lw=0.4)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel("End-to-end latency (s)")
        ax.set_xlabel("Policy and Poisson arrival rate λ")
        legend_above(ax, 2)
        fig.tight_layout()
        save(fig, "fig8b_online_latency")


def fig_robustness() -> None:
    d = load("exp5_summary")
    if not d:
        return
    conds = [c for c, v in d["conditions"].items() if "f1" in v]
    nice = [c.replace("original", "original").replace(":", " ") for c in conds]
    m = np.array([d["conditions"][c]["f1"]["mean"] for c in conds])
    lo = m - np.array([d["conditions"][c]["f1"]["ci95_lo"] for c in conds])
    hi = np.array([d["conditions"][c]["f1"]["ci95_hi"] for c in conds]) - m
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.75))
    x = np.arange(len(conds))
    kind_col = {"original": C["gray"], "blur": C["blue"], "downscale": C["sky"], "lowlight": C["purple"],
                "occlusion": C["red"], "jpeg": C["green"]}
    kind_hatch = {"original": "", "blur": "", "downscale": "///", "lowlight": "...", "occlusion": "xx", "jpeg": "\\\\\\"}
    kinds = [c.split(":")[0] for c in conds]
    ax = axs[0]
    for xi, mi, k, l_, h_ in zip(x, m, kinds, lo, hi):
        ax.bar(xi, mi, yerr=[[l_], [h_]], color=kind_col[k], hatch=kind_hatch[k], capsize=2, edgecolor="black", lw=0.4,
               error_kw={"lw": 0.8})
    ax.axhline(m[0], color=C["gray"], ls=":", lw=0.8)
    ax.set_xticks(x, nice, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Object F1 (mean per image)")
    ax.set_ylim(0, 0.8)
    panel(ax, "a")
    ax = axs[1]
    qf = [d["conditions"][c]["quality_flag_rate"] for c in conds]
    vr = [d["conditions"][c]["vlm_reports_degradation"] for c in conds]
    ax.bar(x - 0.2, qf, 0.4, color=C["orange"], edgecolor="black", lw=0.4, label="image-quality flag raised")
    ax.bar(x + 0.2, vr, 0.4, color=C["sky"], hatch="///", edgecolor="black", lw=0.4, label="model reports degradation")
    ax.set_xticks(x, nice, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of images")
    ax.set_ylim(0, 1.05)
    legend_above(ax, 2, 6.5)
    panel(ax, "b")
    fig.tight_layout(w_pad=2)
    save(fig, "fig9_robustness")


def fig_calibration() -> None:
    d = load("exp6_summary")
    if not d:
        return
    sig = d["object"]["signals"]
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.6))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], color=C["gray"], lw=0.8, ls=":", label="perfect calibration")
    for i, (nm, col) in enumerate((("token prob (raw)", C["orange"]), ("verbal (raw map)", C["red"]),
                                   ("combined (LR, dev)", C["green"]))):
        rel = [b for b in sig[nm]["reliability"] if b["n"] >= 5]
        ax.plot([b["conf"] for b in rel], [b["acc"] for b in rel], marker=MARK[i], ms=4, ls=LS[i], lw=1, color=col, label=nm)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (bin mean)")
    ax.set_ylabel("Observed precision")
    legend_above(ax, 2, 6.5)
    panel(ax, "a")
    ax = axs[1]
    for i, (nm, col) in enumerate((("token prob (raw)", C["orange"]), ("verbal (raw map)", C["red"]),
                                   ("CLIP (Platt, dev)", C["blue"]), ("combined (LR, dev)", C["green"]))):
        rc = sig[nm]["rc_curve"]
        ax.plot(rc["coverage"], rc["risk"], lw=1.1, ls=LS[i], color=col, label=f"{nm}, AURC {sig[nm]['aurc']:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (error rate)")
    legend_above(ax, 2, 6)
    panel(ax, "b")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_calibration")


def fig_ablation() -> None:
    d = load("exp7_partial_summary")
    e3 = load("exp3_summary")
    if not d:
        return
    names, f1o, f1c = [], [], []
    for nm, key in (("Full framework", "full"), ("w/o uncertainty selection", "without_uncertainty"),
                    ("w/o structured validation", "without_validation"), ("w/o CLIP signal", "selector_combined_no_ground"),
                    ("w/o quality signal", "selector_combined_no_quality")):
        if key in d:
            names.append(nm)
            f1o.append(d[key]["objects"]["f1"])
            f1c.append(d[key]["clothing"]["f1"])
    fig, axs = plt.subplots(1, 2 if e3 else 1, figsize=(6.85 if e3 else 3.31, 2.9))
    ax = axs[0] if e3 else axs
    y = np.arange(len(names))
    ax.barh(y - 0.2, f1o, 0.4, color=C["blue"], label="objects", edgecolor="black", lw=0.4)
    ax.barh(y + 0.2, f1c, 0.4, color=C["orange"], hatch="///", label="clothing", edgecolor="black", lw=0.4)
    for yy, v in zip(y - 0.2, f1o):
        ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=6.5)
    for yy, v in zip(y + 0.2, f1c):
        ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y, names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Micro-F1 (test split)")
    ax.set_xlim(0, 1)
    legend_above(ax, 2)
    panel(ax, "a")
    if e3:
        order = ["C2 batched independent (per view, mean)", "C2 independent views (union)",
                 "A  independent + structured aggregation", "C3 contextual multi-image", "C4 contextual + structured aggregation"]
        short = ["Independent, per view", "Independent, union", "Aggregation (A)", "Contextual (C3)", "Contextual + aggr. (C4)"]
        conds = [(o, s) for o, s in zip(order, short) if o in e3["conditions"] and "f1" in e3["conditions"][o]]
        ax = axs[1]
        y = np.arange(len(conds))
        ax.barh(y - 0.2, [e3["conditions"][o]["f1"]["mean"] for o, _ in conds], 0.4, color=C["green"], label="group F1",
                edgecolor="black", lw=0.4)
        ax.barh(y + 0.2, [e3["conditions"][o]["hall"]["mean"] for o, _ in conds], 0.4, color=C["red"], hatch="...",
                label="hallucination rate", edgecolor="black", lw=0.4)
        for yy, v in zip(y - 0.2, [e3["conditions"][o]["f1"]["mean"] for o, _ in conds]):
            ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=6.5)
        for yy, v in zip(y + 0.2, [e3["conditions"][o]["hall"]["mean"] for o, _ in conds]):
            ax.text(v + 0.01, yy, f"{v:.3f}", va="center", fontsize=6.5)
        ax.set_yticks(y, [s for _, s in conds], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Multi-image groups (test)")
        legend_above(ax, 2)
        panel(ax, "b")
    fig.tight_layout(w_pad=2)
    save(fig, "fig10_ablation")


def fig_review_curve() -> None:
    d = load("exp8_summary")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(3.31, 2.7))
    ax.plot(d["curve"]["x"], d["curve"]["ranked"], color=C["green"], lw=1.2, label=f"uncertainty-ranked (AUC {d['auc_ranked']:.2f})")
    ax.plot(d["curve"]["x"], d["curve"]["random"], color=C["gray"], lw=1, ls="--", label=f"random order (AUC {d['auc_random']:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Fraction of observations reviewed")
    ax.set_ylabel("Fraction of errors found")
    legend_above(ax, 1)
    fig.tight_layout()
    save(fig, "fig_review_simulation")


if __name__ == "__main__":
    diagrams()
    fig_throughput_latency()
    fig_robustness()
    fig_calibration()
    fig_ablation()
    fig_review_curve()
