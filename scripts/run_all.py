"""Run the full EDA pipeline as plain scripts (CI-friendly alternative to the
notebook). Produces all tables, figures, artefacts, MODELLING_DECISIONS.md and
DATASET_ANALYSIS_REPORT.md.

Usage:  PYTHONPATH=. python scripts/run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from health_eda import decisions as dec
from health_eda import report

import scripts.run_m1 as m1
import scripts.run_m2 as m2
import scripts.run_m3 as m3
import scripts.run_m4 as m4


def main():
    dec.reset()          # start from a clean decisions log
    m1.main()            # m1 also calls dec.reset(); order preserved
    m2.main()
    m3.main()
    m4.main()
    text = report.build_report()
    print(f"\nWrote report ({len(text):,} chars) and MODELLING_DECISIONS.md")


if __name__ == "__main__":
    main()
