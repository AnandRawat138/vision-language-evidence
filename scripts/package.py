"""Final packaging.

    python scripts/package.py

Creates in deliverables/:
  JRTIP_LaTeX_source.zip          flat LaTeX source (no subfolders, as required by JRTIP): main.tex with all
                                  sections/tables inlined, figures (PDF), svjour3 class files, bib/bbl, compiled PDF
  manuscript_JRTIP.pdf            compiled journal version (svjour3, two-column)
  manuscript_JRTIP.docx           Word version in the JRTIP Word-template layout
  manuscript_journal_neutral.pdf  single-column journal-neutral version
  supplementary_OnlineResource1.pdf
  cover_letter_JRTIP.docx/.pdf      built by paper/make_cover_letter.py
  research_project_complete.zip   code, paper sources, experiment scripts, configs, docs, manifests, derived
                                  references, raw outputs, processed results, logs (no datasets, no model weights)
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "deliverables"

EXCLUDE_DIRS = {".venv", "node_modules", "__pycache__", ".pytest_cache", ".git", "deliverables", "raw", "images",
                "cache", "figures_png", "_template_download"}
EXCLUDE_FILES = {"evidence.sqlite"}
EXCLUDE_SUFFIX = {".aux", ".log", ".out", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz", ".pyc", ".zip"}


def expand(text: str, base: Path) -> str:
    def rep(m: re.Match) -> str:
        p = base / m.group(1)
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        return expand(p.read_text(encoding="utf-8"), base) if p.exists() else ""
    return re.sub(r"\\input\{([^}]+)\}", rep, text)


def latex_zip() -> Path:
    flat = expand((PAPER / "main.tex").read_text(encoding="utf-8"), PAPER)
    flat = flat.replace(r"\graphicspath{{figures/}}", "")
    used = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", flat)))
    z = OUT / "JRTIP_LaTeX_source.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", flat)
        for f in ("svjour3.cls", "svglov3.clo", "spmpsci.bst", "references.bib", "main.bbl"):
            if (PAPER / f).exists():
                zf.write(PAPER / f, f)
        for g in used:
            src = PAPER / "figures" / (g if g.endswith(".pdf") else g + ".pdf")
            zf.write(src, src.name)
        zf.write(PAPER / "main.pdf", "main.pdf")
        zf.writestr("README.txt", "Compile: pdflatex main; bibtex main; pdflatex main; pdflatex main\n"
                                  "All files are in one folder as required by the journal. Class: svjour3 [twocolumn], "
                                  "bibliography style spmpsci (numbered).\n")
    return z


def project_zip() -> Path:
    z = OUT / "research_project_complete.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            parts = set(rel.parts[:-1]) if p.is_file() else set(rel.parts)
            if p.is_dir():
                continue
            # keep results/raw (experiment outputs) and data/annotations; drop dataset images/raw archives
            if rel.parts[:2] == ("data", "raw") or rel.parts[:2] == ("data", "images"):
                continue
            skip_dirs = parts & (EXCLUDE_DIRS - {"raw"})
            if skip_dirs:
                continue
            if p.name in EXCLUDE_FILES or p.suffix in EXCLUDE_SUFFIX:
                continue
            zf.write(p, Path("research_project_complete") / rel)
    return z


def main() -> None:
    OUT.mkdir(exist_ok=True)
    copies = {"main.pdf": "manuscript_JRTIP.pdf", "manuscript_JRTIP.docx": "manuscript_JRTIP.docx",
              "main_neutral.pdf": "manuscript_journal_neutral.pdf", "supplementary.pdf": "supplementary_OnlineResource1.pdf"}
    for s, d in copies.items():
        if (PAPER / s).exists():
            shutil.copy2(PAPER / s, OUT / d)
    a = latex_zip()
    b = project_zip()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:40s} {f.stat().st_size / 1e6:8.2f} MB")
    with zipfile.ZipFile(a) as zf:
        print("LaTeX zip entries:", len(zf.namelist()))
    with zipfile.ZipFile(b) as zf:
        print("project zip entries:", len(zf.namelist()))


if __name__ == "__main__":
    main()
