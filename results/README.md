# Saved results

The frozen final method is **A1: DINOv2-S/14 + residual MLP32, 16-shot, illumination consistency plus 0.1 × clean-feature anchor**. The later capacity, CNN and loss studies are ablations. They do not replace the final method.

## Evaluation tables

Each row in `metrics.csv` is one method × category × seed × test condition. All seven-condition studies use seeds 0, 1 and 2. The seven conditions are clean, brightness 0.70/1.30, contrast 0.70/1.30 and gamma 0.70/1.40. Every selected category uses its complete official test set.

| Directory under `metrics/` | Scope | Methods | Evaluation rows |
|---|---|---|---:|
| `adapter_full_20261102` | Original three categories: bottle, cable, carpet | A0, A1 | 126 |
| `mvtec_extension_20261103` | Full 15 categories; includes the original three controls | A0, A1 | 630 |
| `anchor_ablation_20261102` | Original three categories | A0, A1, A1-no-anchor | 189 |
| `a1_capacity_data_cnn_20261111` | Full 15 categories | A0-16, A0-32, MLP32-16, MLP64-16, MLP32-32, MLP64-32, CNN1-32, CNN3-32 | 2,520 |
| `a1_loss_study_20261111` | Full 15 categories | A0, A1, L1 (anchor 0.2), L2 (fixed clean teacher) | 1,260 |
| `a1_expanded_mlp768_20261116` | Original three categories | A1, MLP768-16 | 126 |
| `a1_expanded_mlp768_full_20261116` | Full 15 categories; includes the three-category stage | A1, MLP768-16 | 630 |
| `defect_adapter_20261102` | Cable pilot, three conditions: clean, brightness 0.70, gamma 1.40 | A0–A3 | 36 |

Controls occur in multiple directories because later studies reuse verified prior results. **Do not add row counts across directories as independent measurements.** The full MLP768 table supersedes the initial three-category table for overall conclusions; the earlier directory retains that stage's diagnostic and paired timing records. The original three categories were the development group. The other twelve were initially an extension group; subsequent studies had already evaluated all fifteen and do not constitute evaluation on entirely unseen categories.

## Reading and reproducing the summaries

- `config.json` records the corresponding fixed experiment settings.
- `metrics.csv` contains all evaluation cells, including decreases and ties. `paired_conditions.csv`, where present, stores differences for the same category, seed and condition. Additional category, condition and seed tables expose heterogeneity beyond the overall mean.
- `summary.json` contains the saved study summary. `summary.md`, where supplied by the original analysis, is its readable counterpart.
- `splits.json` and `batch_manifests.json`, where present, record exact ordered image roles and sampling identities. The capacity study's split manifest covers both nested 16-shot and 32-shot subsets across all fifteen categories. Standalone center-schedule identities are in `../assets/center_schedules.json`.
- `diagnostics*.csv` record feature diagnostics. `costs.csv`, `benchmark_costs.csv`, `benchmark_pairs.csv` and `paired_benchmark.json`, where present, retain measured costs and paired microbenchmarks. Consult each field's units and timing scope: cached-feature scoring is not end-to-end image inference.
- `verification.json`, `pairing_verification.json` and `independent_verification.json`, where present, record the original completeness, identity and numerical audit outcomes. Raw patch-score arrays are not distributed, so their independent metric recomputation requires regenerating those arrays.
- `checkpoint_identities.json` is a compact cross-study index of preserved checkpoint hashes, final-step identity and recorded training costs. Reused checkpoints remain explicitly linked to their original source records.

The primary summary is pixel average precision (AP) across the **six changed-lighting conditions**. Compute paired differences before aggregation, give categories equal weight and conditions equal weight, then summarize the three paired seed values using their mean and sample standard deviation. Report clean-lighting AP and image AUROC separately. A category improves above +0.1 percentage points, declines below −0.1 points, and otherwise ties. The seeds share the same test images; a positive mean does not establish statistical significance.

See the root README for the portable reproduction and saved-table analysis commands. [record_manifest.json](record_manifest.json) and [../PUBLICATION.md](../PUBLICATION.md) describe exact file provenance and the limited public metadata cleanup.
