"""Supplementary table of automated software tests, generated from the JUnit XML of the last pytest run."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vle.reporting.tables import esc, write_table  # noqa: E402

AREA = {"test_core": "core modules", "test_scheduler": "batch scheduler", "test_api": "review API"}


def main() -> None:
    root = ET.parse(ROOT / "results" / "logs" / "pytest_junit.xml").getroot()
    groups: dict[str, list] = defaultdict(list)
    for tc in root.iter("testcase"):
        mod = tc.get("classname", "").split(".")[-1]
        name = tc.get("name", "")
        failed = tc.find("failure") is not None or tc.find("error") is not None
        skipped = tc.find("skipped") is not None
        groups[mod].append((name, "fail" if failed else "skip" if skipped else "pass", float(tc.get("time", 0))))
    rows = []
    for mod, tcs in groups.items():
        for name, st, t in tcs:
            rows.append([esc(AREA.get(mod, mod)), r"\texttt{" + esc(name) + "}", st, f"{t:.2f}"])
    n = sum(len(v) for v in groups.values())
    npass = sum(1 for v in groups.values() for x in v if x[1] == "pass")
    write_table("supp_tests", ["Area", "Test case", "Result", "Time (s)"], rows,
                f"Automated tests executed on the CPU-only configuration ({npass}/{n} passed; pytest, Python 3.12.7). "
                "Covers lexicon mapping, schema coercion, JSON repair, provenance, statement certainty, aggregation, "
                "calibration metrics, statistics, image degradation, de-duplication, scheduler OOM/failure/timeout/resume "
                "handling and API endpoints including the audit trail.", "tab:supp_tests", colspec="llcr", size=r"\scriptsize")
    print(npass, "/", n)


if __name__ == "__main__":
    main()
