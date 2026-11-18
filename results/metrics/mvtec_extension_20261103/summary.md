# Frozen A1: twelve-category MVTec extension

Frozen source: `1ca771cff4cfa770f9356056febf8631cea3dc56`. All 504 new evaluations / 57,750 new image scores audited; 126 audited development evaluations reused.
No method, loss weight, rank, checkpoint rule, training schedule or evaluation setting was selected from these new test outcomes.

The primary analysis is the twelve new categories. The old three development categories and all fifteen combined are separate secondary context.

| Scope | Method | Clean pixel AP | Six-shift pixel AP | Clean I-AUROC | Six-shift I-AUROC |
|---|---|---:|---:|---:|---:|
| new12_primary | A0 | 54.983 +/- 0.533 | 54.396 +/- 0.559 | 98.193 +/- 0.180 | 98.090 +/- 0.189 |
| new12_primary | A1 | 55.846 +/- 0.613 | 55.579 +/- 0.626 | 98.347 +/- 0.175 | 98.366 +/- 0.203 |
| new12_primary | A1 minus A0 (pp) | +0.863 +/- 0.228 | +1.183 +/- 0.249 | +0.154 +/- 0.039 | +0.276 +/- 0.029 |
| old3_development | A0 | 68.303 +/- 0.089 | 68.149 +/- 0.083 | 98.293 +/- 0.226 | 98.343 +/- 0.215 |
| old3_development | A1 | 70.258 +/- 0.381 | 70.123 +/- 0.363 | 98.372 +/- 0.173 | 98.441 +/- 0.148 |
| old3_development | A1 minus A0 (pp) | +1.955 +/- 0.335 | +1.974 +/- 0.344 | +0.079 +/- 0.053 | +0.098 +/- 0.067 |
| all15_secondary | A0 | 57.647 +/- 0.441 | 57.147 +/- 0.458 | 98.213 +/- 0.138 | 98.141 +/- 0.158 |
| all15_secondary | A1 | 58.729 +/- 0.566 | 58.488 +/- 0.574 | 98.352 +/- 0.139 | 98.381 +/- 0.168 |
| all15_secondary | A1 minus A0 (pp) | +1.082 +/- 0.232 | +1.341 +/- 0.249 | +0.139 +/- 0.021 | +0.241 +/- 0.011 |

Each category and condition receives equal weight within each seed. SD is sample SD across three paired reference-sampling seeds, not independent dataset replication. The combined fifteen-category mean weights categories 12:3, not the two scope means 1:1.

| Category | Scope | A0 shift AP | A1 shift AP | Shift gain (pp) | Clean gain (pp) |
|---|---|---:|---:|---:|---:|
| capsule | new | 48.389 | 56.390 | +8.000 +/- 0.332 | +4.721 +/- 0.496 |
| grid | new | 38.272 | 38.496 | +0.224 +/- 1.176 | +0.139 +/- 1.158 |
| hazelnut | new | 82.505 | 82.145 | -0.359 +/- 0.345 | -0.368 +/- 0.361 |
| leather | new | 27.383 | 29.421 | +2.038 +/- 1.233 | +2.030 +/- 1.225 |
| metal_nut | new | 78.167 | 77.856 | -0.311 +/- 0.182 | -0.348 +/- 0.180 |
| pill | new | 69.713 | 69.976 | +0.263 +/- 0.614 | +0.154 +/- 0.593 |
| screw | new | 37.876 | 38.750 | +0.874 +/- 0.142 | +0.387 +/- 0.080 |
| tile | new | 61.208 | 61.727 | +0.519 +/- 0.895 | +0.751 +/- 0.925 |
| toothbrush | new | 62.111 | 62.181 | +0.070 +/- 0.033 | +0.071 +/- 0.015 |
| transistor | new | 41.364 | 42.214 | +0.850 +/- 0.336 | +0.810 +/- 0.333 |
| wood | new | 62.284 | 63.099 | +0.815 +/- 0.359 | +0.824 +/- 0.380 |
| zipper | new | 43.485 | 44.698 | +1.213 +/- 0.268 | +1.189 +/- 0.294 |
| bottle | development | 83.848 | 84.724 | +0.876 +/- 0.142 | +0.888 +/- 0.152 |
| cable | development | 62.369 | 65.357 | +2.988 +/- 0.602 | +2.903 +/- 0.571 |
| carpet | development | 58.228 | 60.288 | +2.059 +/- 0.514 | +2.074 +/- 0.518 |

All 315 category/seed/condition paired changes are in paired_conditions.csv; all 45 category/seed means are in category_seed_pairs.csv. Every regression remains in summary.json and these tables.

## Paired stability and robustness interpretation

- new12_primary: positive shift-AP category means 10/12; category/seed means 27/36; condition pairs 158/216. Shift gain minus clean gain: +0.320 +/- 0.024 pp.
- old3_development: positive shift-AP category means 3/3; category/seed means 9/9; condition pairs 54/54. Shift gain minus clean gain: +0.019 +/- 0.012 pp.
- all15_secondary: positive shift-AP category means 13/15; category/seed means 36/45; condition pairs 212/270. Shift gain minus clean gain: +0.260 +/- 0.021 pp.

Sign counts are descriptive. Improvement on both clean and shifted images alone does not demonstrate a shift-specific robustness advantage.

## Normal-only diagnostics

Four normal images per category/seed are excluded from fitting and retained in the reference bank. Their training-family photometric and synthetic views diagnose behavior; synthetic differences do not certify preservation of real defects. No diagnostic selected checkpoints or changed settings.

| Category | Method | Normal light squared L2 | Anchor squared L2 | Synthetic cross-difference error | Difference norm ratio |
|---|---|---:|---:|---:|---:|
| capsule | A0 | 0.019861 | 0.000000 | 0.030628 | 1.005204 |
| capsule | A1 | 0.005525 | 0.024122 | 0.068630 | 0.882921 |
| grid | A0 | 0.001049 | 0.000000 | 0.004007 | 1.000156 |
| grid | A1 | 0.000754 | 0.004599 | 0.017214 | 0.953591 |
| hazelnut | A0 | 0.000839 | 0.000000 | 0.002117 | 1.002248 |
| hazelnut | A1 | 0.000652 | 0.001791 | 0.005709 | 0.988630 |
| leather | A0 | 0.000859 | 0.000000 | 0.001685 | 1.001126 |
| leather | A1 | 0.000442 | 0.001964 | 0.010872 | 0.969828 |
| metal_nut | A0 | 0.001045 | 0.000000 | 0.001796 | 1.002790 |
| metal_nut | A1 | 0.000778 | 0.002073 | 0.004885 | 0.986727 |
| pill | A0 | 0.003241 | 0.000000 | 0.014421 | 1.012988 |
| pill | A1 | 0.001734 | 0.007193 | 0.024263 | 0.977654 |
| screw | A0 | 0.001190 | 0.000000 | 0.016760 | 1.008264 |
| screw | A1 | 0.000987 | 0.001536 | 0.021244 | 0.986843 |
| tile | A0 | 0.002108 | 0.000000 | 0.010081 | 1.005686 |
| tile | A1 | 0.001284 | 0.004818 | 0.023766 | 0.947138 |
| toothbrush | A0 | 0.001706 | 0.000000 | 0.003686 | 1.002345 |
| toothbrush | A1 | 0.001176 | 0.002645 | 0.010958 | 0.983424 |
| transistor | A0 | 0.000955 | 0.000000 | 0.004700 | 1.002662 |
| transistor | A1 | 0.000809 | 0.001244 | 0.007142 | 0.985215 |
| wood | A0 | 0.001409 | 0.000000 | 0.004267 | 0.999529 |
| wood | A1 | 0.000923 | 0.009655 | 0.017755 | 0.963537 |
| zipper | A0 | 0.002073 | 0.000000 | 0.008680 | 1.011273 |
| zipper | A1 | 0.001417 | 0.002755 | 0.019851 | 0.954704 |
| bottle | A0 | 0.003203 | 0.000000 | 0.005595 | 1.008785 |
| bottle | A1 | 0.001225 | 0.004348 | 0.019877 | 0.952428 |
| cable | A0 | 0.005056 | 0.000000 | 0.009262 | 1.003641 |
| cable | A1 | 0.002594 | 0.009205 | 0.021135 | 0.955744 |
| carpet | A0 | 0.000810 | 0.000000 | 0.002320 | 1.000789 |
| carpet | A1 | 0.000549 | 0.001053 | 0.007912 | 0.942939 |

Per-seed diagnostics, including directional cosine, are in diagnostics.csv.

## Audit and resources

The original DefectAdapter and fit_one(A1) are reused without modification. Audits reconstruct initialization and every sampled batch, verify normal-only 12/4 partitions, final step 1000, frozen artifact hashes, A1 banks and diagnostics. Every new AP is recomputed from saved patches, every AUROC independently pair-counted; clean seed-0 AP is checked with sklearn and 32 first-image query patches per condition with CPU FAISS.
The pre-training old-category bridge is stored in preflight.json. Reused development metrics retain their original audit and hashes.
Audit maximum errors: `{"bank": 0.0, "diagnostics": 0.0, "faiss": 1.0468065738677979e-06, "independent_auroc": 1.1102230246251565e-16, "patch_to_image": 0.0, "reconstructed_ap": 0.0, "zero_adapter_feature": 8.940696716308594e-08}`.

New normal cache extraction and fitting: 38.01 min; evaluation: 102.43 min; audit: 40.42 min.
Resource counters: `{"eval_resources": {"peak_cuda_allocated_mib": 1158.09228515625, "peak_cuda_reserved_mib": 1384.0, "process_lifetime_peak_ram_mib": 3996.79296875}, "fit_resources": {"peak_cuda_allocated_mib": 193.96630859375, "peak_cuda_reserved_mib": 222.0, "process_lifetime_peak_ram_mib": 2225.3515625}}`.

This experiment tests transfer of the frozen A1 recipe within MVTec AD. It does not establish cross-dataset generalization, statistical independence of seeds, or SOTA. All outcomes are retained; no gate triggers tuning on these test results.
