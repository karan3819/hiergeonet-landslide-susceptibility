#!/usr/bin/env python3
"""
Cross-checks results/metrics/test_metrics.json (the LEGACY single-seed test
set result) against the confusion matrix it corresponds to, so a hand-edit
or copy-paste error would be caught automatically rather than silently
shipping as a "reference" number.

This check only applies to the legacy single-run result. It does NOT apply
to results/metrics/benchmark_comparison_multiseed.csv (the current 6-seed
result) — see the comment in .github/workflows/integrity-check.yml for why
an analogous check on mean-of-6-seeds values would produce false positives
for models with genuine cross-seed variance.

Checks performed:
  1. test_metrics.json's n_samples and pos_rate are recomputable from the
     confusion matrix pixel counts hardcoded below (sourced from
     results/figures/roc_confusion.png, the legacy confusion matrix figure).
  2. Precision, Recall, and F1 in test_metrics.json match what those same
     confusion matrix counts imply, to 1e-6 tolerance.

This script intentionally does NOT re-run the model. It only checks
consistency between numbers that are already committed to the repo.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "results" / "metrics" / "test_metrics.json"

# Pixel counts read off the confusion matrix in results/figures/roc_confusion.png
# (No LS = negative class, LS = positive/landslide class)
TN = 12_732_762  # predicted No LS, actually No LS
FP = 75_625      # predicted LS,    actually No LS
FN = 74_769      # predicted No LS, actually LS
TP = 224_044     # predicted LS,    actually LS

TOL = 1e-6


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"FAIL: {METRICS_PATH} not found")
        return 1

    metrics = json.loads(METRICS_PATH.read_text())

    total = TN + FP + FN + TP
    pos = FN + TP
    pos_rate = pos / total

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    checks = [
        ("n_samples", total, metrics["n_samples"], 0),
        ("pos_rate", pos_rate, metrics["pos_rate"], TOL),
        ("Precision", precision, metrics["Precision"], TOL),
        ("Recall", recall, metrics["Recall"], TOL),
        ("F1", f1, metrics["F1"], TOL),
    ]

    ok = True
    for name, computed, reported, tol in checks:
        diff = abs(computed - reported)
        status = "OK" if diff <= tol else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"[{status}] {name}: computed={computed!r} reported={reported!r} diff={diff!r}")

    if not ok:
        print("\nOne or more consistency checks failed.")
        return 1

    print("\nAll consistency checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
