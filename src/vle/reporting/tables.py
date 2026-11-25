"""Generate LaTeX (booktabs) tables directly from processed result files."""

from __future__ import annotations

from pathlib import Path

from vle.runio import ROOT

TABLE_DIR = ROOT / "results" / "tables"


def fmt(v, nd: int = 3) -> str:  # type: ignore[no-untyped-def]
    if v is None:
        return "--"
    if isinstance(v, float):
        if v != v:  # nan
            return "--"
        return f"{v:.{nd}f}"
    return str(v)


def ci(m: float | None, lo: float | None, hi: float | None, nd: int = 3) -> str:
    if m is None or m != m:
        return "--"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def write_table(name: str, header: list[str], rows: list[list[str]], caption: str, label: str,
                colspec: str | None = None, note: str | None = None, size: str = r"\footnotesize",
                wide: bool = True) -> Path:
    """Booktabs table. Wide tables span both columns (table*); every tabular is wrapped in
    adjustbox{max width=...} so it can never exceed the text/column width."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    colspec = colspec or ("l" + "r" * (len(header) - 1))
    env = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{adjustbox}}{{max width={width}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        if r == ["MIDRULE"]:
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    if note:
        lines.append(rf"\par\smallskip\parbox{{0.95{width}}}{{\footnotesize {note}}}")
    lines.append(rf"\end{{{env}}}")
    p = TABLE_DIR / f"{name}.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paper_tables = ROOT / "paper" / "tables"
    paper_tables.mkdir(parents=True, exist_ok=True)
    (paper_tables / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
