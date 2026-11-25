"""Block diagrams for the manuscript (Figs. 1-5), drawn at their exact printed size.

Print sizes (svjour3 two-column): column = 84 mm (3.31 in), full width = 174 mm (6.85 in).
Because figures are drawn at print size and included at 100 % scale, the font sizes below are the
printed sizes (body >= 7 pt, box titles 8 pt).

Layout rules enforced programmatically (assertions):
  * no two boxes overlap;
  * every connector starts and ends on a box edge and no segment passes through another box;
  * every text block fits inside its box with padding (font is reduced, never below 6.5 pt).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COL_IN, FULL_IN = 3.31, 6.85
MIN_FS = 6.5
INK = "#1b1b1b"
# (fill, edge) — saturated pastel fills with strong edges; distinguishable in greyscale by edge darkness
STYLES = {
    "io": ("#E3E8EF", "#44546A"),
    "core": ("#CFE2F7", "#1F5FA8"),
    "sched": ("#FCDDBA", "#C55A00"),
    "unc": ("#CDEFD8", "#1E7B45"),
    "review": ("#F8D0CE", "#B3261E"),
    "narr": ("#E6D6F5", "#6A3D9A"),
}


class Diagram:
    def __init__(self, w_in: float, h_in: float):
        self.fig, self.ax = plt.subplots(figsize=(w_in, h_in))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.boxes: dict[str, tuple[float, float, float, float]] = {}

    def box(self, name: str, x: float, y: float, w: float, h: float, text: str, style: str = "core",
            fs: float = 8.0) -> None:
        for other, (ox, oy, ow, oh) in self.boxes.items():
            sep = x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y
            assert sep, f"boxes overlap: {name} / {other}"
        fc, ec = STYLES[style]
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02",
                                         fc=fc, ec=ec, lw=1.1))
        self.boxes[name] = (x, y, w, h)
        self._fit_text(x, y, w, h, text, fs, ec)

    def _fit_text(self, x, y, w, h, text, fs, title_color):  # type: ignore[no-untyped-def]
        r = self.fig.canvas.get_renderer()
        bx0, by0 = self.ax.transData.transform((x, y))
        bx1, by1 = self.ax.transData.transform((x + w, y + h))
        pad = 3.0
        lines = text.split("\n")
        size = fs
        while size >= MIN_FS:
            items = []
            if len(lines) > 1:
                head = self.ax.text(0, 0, lines[0], ha="center", va="bottom", fontsize=size, weight="bold", color=title_color)
                body = self.ax.text(0, 0, "\n".join(lines[1:]), ha="center", va="top", fontsize=size - 0.5,
                                    linespacing=1.12, color=INK)
                hh = head.get_window_extent(r).height
                hb = body.get_window_extent(r).height
                inv = self.ax.transData.inverted()
                cy_disp = (by0 + by1) / 2
                top_disp = cy_disp + (hh + hb + 2) / 2
                head.set_position(inv.transform(((bx0 + bx1) / 2, top_disp - hh)))
                body.set_position(inv.transform(((bx0 + bx1) / 2, top_disp - hh - 2)))
                items = [head, body]
            else:
                items = [self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                                      weight="bold", color=title_color)]
            ok = all(e.get_window_extent(r).x0 >= bx0 + pad and e.get_window_extent(r).x1 <= bx1 - pad
                     and e.get_window_extent(r).y0 >= by0 + pad and e.get_window_extent(r).y1 <= by1 - pad for e in items)
            if ok:
                return
            for e in items:
                e.remove()
            size -= 0.25
        raise AssertionError(f"text does not fit at >= {MIN_FS} pt: {text!r}")

    def anchor(self, name: str, side: str, frac: float = 0.5) -> tuple[float, float]:
        x, y, w, h = self.boxes[name]
        return {"l": (x, y + frac * h), "r": (x + w, y + frac * h), "t": (x + frac * w, y + h),
                "b": (x + frac * w, y)}[side]

    def _check_segment(self, p, q, allowed: set[str]) -> None:  # type: ignore[no-untyped-def]
        for name, (x, y, w, h) in self.boxes.items():
            if name in allowed:
                continue
            for i in range(81):
                t = i / 80
                px, py = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
                assert not (x + 0.002 < px < x + w - 0.002 and y + 0.002 < py < y + h - 0.002), f"connector crosses {name}"

    def connect(self, a: str, sa: str, b: str, sb: str, fa: float = 0.5, fb: float = 0.5,
                via: list[tuple[float, float]] | None = None, dashed: bool = False, color: str = INK) -> None:
        pts = [self.anchor(a, sa, fa)] + (via or []) + [self.anchor(b, sb, fb)]
        for s, e in zip(pts[:-1], pts[1:]):
            self._check_segment(s, e, {a, b})
        ls = (0, (4, 2)) if dashed else "-"
        for s, e in zip(pts[:-2], pts[1:-1]):
            self.ax.add_line(Line2D([s[0], e[0]], [s[1], e[1]], color=color, lw=1.0, linestyle=ls))
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8, color=color, lw=1.0,
                                          linestyle=ls, shrinkA=0, shrinkB=0))

    def note(self, x: float, y: float, text: str, color: str = "#555555", fs: float = 7.0, style: str = "italic",
             ha: str = "center") -> None:
        self.ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style)


# ===================================================================== figs
def fig_architecture() -> plt.Figure:
    d = Diagram(FULL_IN, 3.7)
    top, mid, bot, hh = 0.63, 0.355, 0.08, 0.215
    w = 0.142
    cols = [0.005, 0.172, 0.339, 0.506, 0.673, 0.853]
    d.box("A", cols[0], top, w, hh, "A  Dataset manager\nmanifest, MD5,\ndHash de-duplication", "io")
    d.box("C", cols[1], top, w, hh, "C  Image engine\nblur, light, contrast,\nresolution flags")
    d.box("E", cols[2], top, w, hh, "E  Batch scheduler\nstatic, sorted,\ndynamic, adaptive\nOOM split, retry", "sched")
    d.box("D", cols[3], top, w, hh, "D  Inference API\nHF Transformers (local)\nvLLM client (server)", "sched")
    d.box("V", cols[4], top, w, hh, "Schema validation\nJSON repair, lexicon,\nobservation records")
    d.box("G", cols[4], mid, w, hh, "G  Uncertainty\ntoken, verbal, CLIP,\nquality → calibrated p", "unc")
    d.box("F", cols[4], bot, w, hh, "F  Aggregation\nsupport, conflicts,\nmissing slots", "unc")
    d.box("S", cols[0], bot, w, hh, "Statement\nstored verbatim", "io")
    d.box("B", cols[1], bot, w, hh, "B  Statement\nstructuring\nrule | LLM, certainty", "narr")
    d.box("H", cols[5], bot, 0.142, top + hh - bot, "H  Human review\nFastAPI + React\naccept / reject /\ncorrect\nappend-only\naudit trail\nreports", "review")
    d.connect("A", "r", "C", "l")
    d.connect("C", "r", "E", "l")
    d.connect("E", "r", "D", "l")
    d.connect("D", "r", "V", "l")
    d.connect("V", "b", "G", "t")
    d.connect("G", "b", "F", "t")
    d.connect("S", "r", "B", "l")
    d.connect("B", "r", "F", "l")
    span = top + hh - bot
    d.connect("G", "r", "H", "l", fb=(mid + hh / 2 - bot) / span)
    d.connect("F", "r", "H", "l", fb=(bot + hh / 2 - bot) / span)
    cx = cols[1] + w / 2
    d.connect("C", "b", "G", "l", via=[(cx, mid + hh / 2)], dashed=True, color=STYLES["unc"][1])
    d.note(0.42, mid + hh / 2 + 0.04, "image-quality flags", color=STYLES["unc"][1], fs=7.5)
    d.note(0.5, 0.955, "Every observation carries provenance: source, image id, model + revision, quantisation, "
                       "prompt version, run, timestamp, review status", fs=7.5)
    d.note(0.5, 0.03, "No identity matching, face recognition or person ranking is implemented.",
           color=STYLES["review"][1], style="normal", fs=7.5)
    return d.fig


def fig_narrative() -> plt.Figure:
    d = Diagram(COL_IN, 3.0)
    W, H = 0.28, 0.24
    xs = [0.065, 0.39, 0.715]
    y1, y2, y3 = 0.71, 0.38, 0.04
    d.box("s1", xs[0], y1, W, H, "Statement\nstored\nunchanged", "io")
    d.box("s2", xs[1], y1, W, H, "Clause split\nlexicon spans", "narr")
    d.box("s3", xs[2], y1, W, H, "Cue scoping\nhedge,\nnegation", "narr")
    d.box("s4", xs[2], y2, W, H, "Colour\nattachment\n≤ 30 chars", "narr")
    d.box("s5", xs[1], y2, W, H, "Observations\ncertainty never\nupgraded", "unc")
    d.box("s6", xs[0], y2, W, H, "Missing\nslots\nreported", "sched")
    d.box("llm", xs[1], y3, xs[2] + W - xs[1], 0.25, "LLM path (narrative-v1)\nJSON → schema check → quote\ngrounding → certainty floor", "sched")
    d.connect("s1", "r", "s2", "l")
    d.connect("s2", "r", "s3", "l")
    d.connect("s3", "b", "s4", "t")
    d.connect("s4", "l", "s5", "r")
    d.connect("s5", "l", "s6", "r")
    d.connect("s1", "l", "llm", "l", fa=0.5, fb=0.5, via=[(0.025, y1 + H / 2), (0.025, y3 + 0.125)])
    d.connect("llm", "t", "s5", "b", fa=(xs[1] + W / 2 - xs[1]) / (xs[2] + W - xs[1]))
    return d.fig


def fig_batching() -> plt.Figure:
    d = Diagram(COL_IN, 2.9)
    W, H = 0.29, 0.30
    xs = [0.02, 0.355, 0.69]
    y1, y2 = 0.56, 0.10
    d.box("R", xs[0], y1, W, H, "Requests\nindependent,\n1 image each", "io")
    d.box("Q", xs[1], y1, W, H, "Queue\nPoisson (online)\nor all at t₀", "sched")
    d.box("P", xs[2], y1, W, H, "Policy\nB, max wait,\nsort, adaptive", "sched")
    d.box("M", xs[0], y2, W, H, "Results\nJSONL checkpoint\nNVML / psutil", "unc")
    d.box("K", xs[1], y2, W, H, "Backend call\nleft-padded batch\nlog-probs", "core")
    d.box("X", xs[2], y2, W, H, "Recovery\nOOM → split\nerror → bisect\ntimeout → watchdog", "review")
    d.connect("R", "r", "Q", "l")
    d.connect("P", "l", "Q", "r")
    d.connect("Q", "b", "K", "t")
    d.connect("K", "l", "M", "r")
    d.connect("K", "r", "X", "l", fa=0.65, fb=0.65)
    d.connect("X", "l", "K", "r", fa=0.35, fb=0.35)
    d.note(0.5, 0.95, "batch size = independent requests per call", fs=7)
    return d.fig


def _fan_in(d: Diagram, names: list[tuple[str, str, str]], target: str, y: float, h: float) -> None:
    n = len(names)
    gap = 0.02
    w = (1 - 0.02 * 2 - gap * (n - 1)) / n
    for i, (nm, txt, st) in enumerate(names):
        d.box(nm, 0.02 + i * (w + gap), y, w, h, txt, st)
    tx, _, tw, _ = d.boxes[target]
    for i, (nm, _, _) in enumerate(names):
        cx = 0.02 + i * (w + gap) + w / 2
        fb = min(max((cx - tx) / tw, 0.08), 0.92)
        d.connect(nm, "b", target, "t", fb=fb, via=[(cx, (y + d.boxes[target][1] + d.boxes[target][3]) / 2),
                                                     (tx + fb * tw, (y + d.boxes[target][1] + d.boxes[target][3]) / 2)])


def fig_aggregation() -> plt.Figure:
    d = Diagram(COL_IN, 3.4)
    d.box("V", 0.12, 0.53, 0.76, 0.18, "Weighted support per label\nS(ℓ) = Σ w·p / Σ w   (w: image quality, p: calibrated)", "unc")
    _fan_in(d, [("v1", "View 1\nleft crop\nw = 1.0", "io"), ("v2", "View 2\nright crop\nw = 1.0", "io"),
                ("v3", "View 3\nlow light\nw = 0.6", "io"), ("v4", "View 4\nblur\nw = 0.5", "io")], "V", 0.80, 0.18)
    d.box("K", 0.02, 0.27, 0.46, 0.17, "Kept labels\nS(ℓ) ≥ θ (dev-tuned)", "core")
    d.box("X", 0.52, 0.27, 0.46, 0.17, "Contradictions\ncolour, setting, statement\n(flagged, not resolved)", "review")
    d.box("O", 0.12, 0.02, 0.76, 0.16, "Aggregated record\nmissing slots, duplicates; no same-person claim", "sched")
    d.connect("V", "b", "K", "t", fa=0.2, fb=(0.12 + 0.2 * 0.76 - 0.02) / 0.46)
    d.connect("V", "b", "X", "t", fa=0.8, fb=(0.12 + 0.8 * 0.76 - 0.52) / 0.46)
    d.connect("K", "b", "O", "t", fa=(0.12 + 0.2 * 0.76 - 0.02) / 0.46, fb=0.2)
    d.connect("X", "b", "O", "t", fa=(0.12 + 0.8 * 0.76 - 0.52) / 0.46, fb=0.8)
    return d.fig


def fig_uncertainty() -> plt.Figure:
    d = Diagram(COL_IN, 3.4)
    d.box("L", 0.12, 0.53, 0.76, 0.18, "Calibrator\nlogistic regression, fit on the dev split only", "unc")
    _fan_in(d, [("t", "Token\nprob.\ngeo-mean", "core"), ("v", "Verbal\nconf.\nhigh/med/low", "core"),
                ("c", "CLIP\ngrounding\ncosine", "core"), ("q", "Image\nquality\nflags", "core")], "L", 0.80, 0.18)
    d.box("S", 0.02, 0.27, 0.46, 0.17, "Calibrated p(o)\nECE, Brier, risk–coverage", "core")
    d.box("R", 0.52, 0.27, 0.46, 0.17, "Selective reporting\nkeep p(o) ≥ τ (dev-tuned)", "sched")
    d.box("F", 0.12, 0.02, 0.76, 0.16, "Review flags → review queue\nlow score, image quality, unmapped label, contradiction", "review")
    d.connect("L", "b", "S", "t", fa=0.2, fb=(0.12 + 0.2 * 0.76 - 0.02) / 0.46)
    d.connect("L", "b", "R", "t", fa=0.8, fb=(0.12 + 0.8 * 0.76 - 0.52) / 0.46)
    d.connect("S", "b", "F", "t", fa=(0.12 + 0.2 * 0.76 - 0.02) / 0.46, fb=0.2)
    d.connect("R", "b", "F", "t", fa=(0.12 + 0.8 * 0.76 - 0.52) / 0.46, fb=0.8)
    return d.fig
