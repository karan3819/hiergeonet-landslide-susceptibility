# Metrics files — data dictionary

All files in this directory are unmodified outputs of the scripts noted per-section below (see repository root README §2.3 for the full current-vs-legacy pipeline distinction).

---

## Current pipeline (`train.py`)

### `benchmark_comparison_multiseed.csv`

Produced by `train.py --aggregate_only` (via `aggregate_seed_results()`), aggregating per-seed JSON files across `Config.SEEDS = [42, 43, 44, 45, 46, 47]`. One row per model; `Model` is the index column. Each model is evaluated on the **held-out test set**, at its own **F1-optimal threshold selected independently per seed** (via a precision-recall sweep on that seed's test-set predictions — see `train_hiergeonet_one_seed()` / `train_gnn_baseline_one_seed()` / `fit_flat_baseline_with_resume()` in `train.py`).

| Column | Meaning |
|---|---|
| `{Metric}_mean` | Mean of that metric across the 6 seeds |
| `{Metric}_std` | Standard deviation of that metric across the 6 seeds (population std, i.e. `numpy.std()` default `ddof=0`) |
| `n_seeds` | Number of seeds successfully aggregated (6 for every model in the committed CSV) |

Metrics covered: `AUC` (ROC-AUC), `F1`, `IoU` (Jaccard index), `Recall`, `Precision`, `MCC` (Matthews correlation coefficient), `Accuracy`. Note that **average precision (AUC-PR) is not computed by the current pipeline** and does not appear in this file — the AUC-PR values shown in `results/figures/fig2_roc_pr_curves.png` come from a separate, single-run computation (see root README §5.2).

**A note on per-seed threshold selection:** because each seed picks its own F1-optimal threshold independently, the `Precision_mean`/`Recall_mean` pair reported here is the mean of six independently-optimized precision/recall pairs, not the precision/recall you'd get by applying one fixed threshold six times. This is a reasonable and common protocol, but worth knowing if you try to reconcile these means against a specific single threshold value elsewhere.

---

## Ablation study (unchanged from legacy pipeline)

### `ablation_study.csv`

Validation set (245 patches). Produced by `ablation_study.py` (root of repo), which is `legacy/run_hiergeonet.py`'s original ablation logic (STAGE 4) extracted into a standalone script — see that script's module docstring for full extraction/provenance notes. This file has **not** been regenerated since the pipeline was updated to `train.py`; it is the same file produced by the legacy pipeline. `Configuration` is the index column; `HierGeoNet (full)` is the reference row.

| Column | Meaning |
|---|---|
| `AUC`, `F1`, `Prec`, `Rec` | Standard metrics at a **fixed 0.5 decision threshold** (see `eval_ablation()` in `ablation_study.py`) |
| `ΔAUC`, `ΔF1` | Difference from the `HierGeoNet (full)` row (`this row − full`) |

**Note on threshold consistency:** the ablation study uses a fixed 0.5 threshold, which is a different convention from both the current pipeline's per-seed F1-optimal threshold and the legacy single-seed pipeline's F1-optimal threshold described below. This is intentional — an ablation study needs one configuration-independent operating point for the comparison to be meaningful — but means ablation numbers are not directly comparable to either benchmark table without accounting for the threshold difference.

**Note on training objective:** the ablation variants are trained with plain weighted BCE only (no Dice loss, no geographic contrastive term), unlike HierGeoNet's main training run in either pipeline version, which combines all three losses. This is a deliberate difference in the original script that was preserved during extraction, not an inconsistency introduced later.

---

## Legacy pipeline (`legacy/run_hiergeonet.py`)

These files were produced by the pipeline's previous, single-seed version and are preserved for anyone cross-referencing an earlier version of the paper. See root README §2.3 for why both versions are kept.

### `test_metrics.json`

Held-out test set (800 patches, 13,107,200 pixels). Computed once, from the best checkpoint by validation AUC, after training is complete, at a single seed (42).

| Field | Meaning |
|---|---|
| `AUC` | ROC-AUC, threshold-independent |
| `F1` | F1 score at `threshold_used` |
| `Recall` | Recall (sensitivity) at `threshold_used` |
| `Precision` | Precision at `threshold_used` |
| `MCC` | Matthews correlation coefficient at `threshold_used` |
| `AP` | Average precision (area under precision-recall curve), threshold-independent |
| `threshold_used` | Decision threshold on predicted probability, chosen to maximize F1 via a precision-recall sweep over the test set |
| `n_samples` | Total pixel count evaluated (800 patches × 128 × 128) |
| `pos_rate` | Fraction of pixels labeled positive (landslide) in the test set |

### `benchmark_comparison.csv`

Validation set (245 patches), single seed. One row per model; `Model` is the index column; includes SGCN-LSTM, which is not present in the current pipeline's comparison.

| Column | Meaning |
|---|---|
| `AUC`, `F1`, `Prec`, `Rec`, `MCC` | Standard binary classification metrics. Classical/deep baselines use a fixed 0.5 decision threshold; **`HierGeoNet (ours)`'s row is the one exception** — it reuses the model's held-out test-set metrics at its F1-optimal threshold from `test_metrics.json` (see the `rows.append({'Model': 'HierGeoNet (ours)', ...})` call in `legacy/run_hiergeonet.py`) rather than being recomputed at 0.5 on the validation set like every other row. This is a real asymmetry in the legacy evaluation protocol, not a copy-paste artifact. |
| `Time_s` | Wall-clock training time in seconds, where measured (classical ML baselines only; blank for neural models) |
