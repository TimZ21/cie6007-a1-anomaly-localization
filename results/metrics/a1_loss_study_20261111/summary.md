# Fixed MLP32/16-shot loss study

Complete matrix: **1,260 evaluations** (630 new + 630 reused), **90 new final-step heads** and 45 reused A1 heads.

A0 is the frozen no-adapter baseline. A1 uses illumination consistency + 0.1 anchor; L1 changes only anchor to 0.2. L2 replaces illumination consistency with the distance from the adapted shifted feature to the frozen clean teacher, retaining 0.1 clean anchor.

Differences are formed within the same category, seed and condition; categories and conditions are equally weighted within each seed. Report mean and sample SD across the three paired seed aggregates.

All categories were previously inspected. The original3/prior12 groups are not newly unseen validation sets. Seeds vary reference selection, initialization and center sampling on one shared test set; they are not independent datasets. Practical gates and positive means are not significance tests.

Scores are percentages; differences are percentage points. ± is sample SD across seeds.

## all15

| Arm | Six-shift pixel AP | Clean pixel AP | Six-shift image AUROC | Clean image AUROC |
| --- | ---: | ---: | ---: | ---: |
| A0 | 57.147 ± 0.458 | 57.647 ± 0.441 | 98.141 ± 0.158 | 98.213 ± 0.138 |
| A1 | 58.488 ± 0.574 | 58.729 ± 0.566 | 98.381 ± 0.168 | 98.352 ± 0.139 |
| L1 | 58.146 ± 0.539 | 58.408 ± 0.534 | 98.352 ± 0.159 | 98.355 ± 0.139 |
| L2 | 57.466 ± 0.446 | 57.786 ± 0.432 | 98.255 ± 0.163 | 98.241 ± 0.148 |

| Paired comparison | Shift AP Δ ± SD | Clean AP Δ ± SD | Shift AUROC Δ ± SD | Clean AUROC Δ ± SD | Categories +/tie/− | Positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1_minus_A1 | -0.342 ± 0.061 | -0.321 ± 0.055 | -0.029 ± 0.010 | +0.003 ± 0.017 | 3/2/10 | 0/3 |
| L2_minus_A1 | -1.022 ± 0.186 | -0.942 ± 0.180 | -0.126 ± 0.020 | -0.111 ± 0.034 | 2/2/11 | 0/3 |
| A1_minus_A0 | +1.341 ± 0.249 | +1.082 ± 0.232 | +0.241 ± 0.011 | +0.139 ± 0.021 | 12/1/2 | 3/3 |
| L1_minus_A0 | +0.999 ± 0.188 | +0.761 ± 0.177 | +0.211 ± 0.008 | +0.142 ± 0.004 | 11/1/3 | 3/3 |
| L2_minus_A0 | +0.320 ± 0.076 | +0.140 ± 0.066 | +0.114 ± 0.015 | +0.028 ± 0.015 | 7/6/2 | 3/3 |
| L2_minus_L1 | -0.680 ± 0.127 | -0.621 ± 0.128 | -0.097 ± 0.023 | -0.114 ± 0.018 | 3/1/11 | 0/3 |

Mean paired six-shift pixel AP over seeds: > +0.1 pp improved, < -0.1 pp degraded, inclusive interval [-0.1,+0.1] tied.

## original3_development

| Arm | Six-shift pixel AP | Clean pixel AP | Six-shift image AUROC | Clean image AUROC |
| --- | ---: | ---: | ---: | ---: |
| A0 | 68.149 ± 0.083 | 68.303 ± 0.089 | 98.343 ± 0.215 | 98.293 ± 0.226 |
| A1 | 70.123 ± 0.363 | 70.258 ± 0.381 | 98.441 ± 0.148 | 98.372 ± 0.173 |
| L1 | 69.651 ± 0.235 | 69.791 ± 0.262 | 98.390 ± 0.145 | 98.336 ± 0.164 |
| L2 | 68.418 ± 0.084 | 68.572 ± 0.101 | 98.342 ± 0.184 | 98.299 ± 0.168 |

| Paired comparison | Shift AP Δ ± SD | Clean AP Δ ± SD | Shift AUROC Δ ± SD | Clean AUROC Δ ± SD | Categories +/tie/− | Positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1_minus_A1 | -0.472 ± 0.128 | -0.467 ± 0.122 | -0.051 ± 0.006 | -0.035 ± 0.010 | 0/0/3 | 0/3 |
| L2_minus_A1 | -1.705 ± 0.306 | -1.687 ± 0.291 | -0.099 ± 0.037 | -0.073 ± 0.031 | 0/0/3 | 0/3 |
| A1_minus_A0 | +1.974 ± 0.344 | +1.955 ± 0.335 | +0.098 ± 0.067 | +0.079 ± 0.053 | 3/0/0 | 3/3 |
| L1_minus_A0 | +1.502 ± 0.225 | +1.488 ± 0.227 | +0.047 ± 0.071 | +0.044 ± 0.062 | 3/0/0 | 3/3 |
| L2_minus_A0 | +0.269 ± 0.040 | +0.268 ± 0.050 | -0.000 ± 0.031 | +0.006 ± 0.063 | 2/1/0 | 3/3 |
| L2_minus_L1 | -1.233 ± 0.185 | -1.220 ± 0.179 | -0.048 ± 0.042 | -0.037 ± 0.027 | 0/0/3 | 0/3 |

Mean paired six-shift pixel AP over seeds: > +0.1 pp improved, < -0.1 pp degraded, inclusive interval [-0.1,+0.1] tied.

## prior12_extension

| Arm | Six-shift pixel AP | Clean pixel AP | Six-shift image AUROC | Clean image AUROC |
| --- | ---: | ---: | ---: | ---: |
| A0 | 54.396 ± 0.559 | 54.983 ± 0.533 | 98.090 ± 0.189 | 98.193 ± 0.180 |
| A1 | 55.579 ± 0.626 | 55.846 ± 0.613 | 98.366 ± 0.203 | 98.347 ± 0.175 |
| L1 | 55.270 ± 0.615 | 55.562 ± 0.603 | 98.342 ± 0.193 | 98.360 ± 0.178 |
| L2 | 54.729 ± 0.539 | 55.090 ± 0.515 | 98.233 ± 0.199 | 98.226 ± 0.187 |

| Paired comparison | Shift AP Δ ± SD | Clean AP Δ ± SD | Shift AUROC Δ ± SD | Clean AUROC Δ ± SD | Categories +/tie/− | Positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L1_minus_A1 | -0.310 ± 0.065 | -0.284 ± 0.062 | -0.024 ± 0.011 | +0.012 ± 0.019 | 3/2/7 | 0/3 |
| L2_minus_A1 | -0.851 ± 0.170 | -0.756 ± 0.166 | -0.133 ± 0.034 | -0.121 ± 0.039 | 2/2/8 | 0/3 |
| A1_minus_A0 | +1.183 ± 0.249 | +0.863 ± 0.228 | +0.276 ± 0.029 | +0.154 ± 0.039 | 9/1/2 | 3/3 |
| L1_minus_A0 | +0.873 ± 0.187 | +0.579 ± 0.171 | +0.252 ± 0.028 | +0.166 ± 0.020 | 8/1/3 | 3/3 |
| L2_minus_A0 | +0.332 ± 0.088 | +0.107 ± 0.074 | +0.143 ± 0.012 | +0.033 ± 0.007 | 5/5/2 | 3/3 |
| L2_minus_L1 | -0.541 ± 0.114 | -0.472 ± 0.116 | -0.109 ± 0.036 | -0.133 ± 0.020 | 3/1/8 | 0/3 |

Mean paired six-shift pixel AP over seeds: > +0.1 pp improved, < -0.1 pp degraded, inclusive interval [-0.1,+0.1] tied.

## Prespecified practical gate

All15 only, compared with A1: shift AP ≥ +0.3 pp; all three shift-AP seed gains > 0; clean AP ≥ −0.2 pp; both clean and shift image AUROC ≥ −0.1 pp; more improved than degraded categories using the ±0.1 pp category band.

| Candidate | Gate | Failed conditions |
| --- | --- | --- |
| L1 | fail | shift_pixel_ap_guardrail, clean_pixel_ap_guardrail, all_three_shift_seed_gains_positive, more_improved_than_degraded_categories |
| L2 | fail | shift_pixel_ap_guardrail, clean_pixel_ap_guardrail, shift_image_auroc_guardrail, clean_image_auroc_guardrail, all_three_shift_seed_gains_positive, more_improved_than_degraded_categories |

Apply the rule to all15 only, using unrounded paired values. A pass meets the predeclared practical criteria; it does not establish statistical significance, novelty, or a reason to overwrite frozen A1. All candidates, including failed gates, remain in the report.

## Common normal-only diagnostics

All four arms use the same four diagnostic normal images, excluded from adapter fitting but included in the reference bank. Diagnostics do not choose settings or checkpoints. The common unweighted quantities are student-to-student light drift, clean feature displacement and distance from the shifted student to the fixed clean teacher. Total losses have different definitions/weights and are not comparable performance scores.

| Arm | Light drift squared L2 | Clean anchor squared L2 | Fixed-clean-teacher squared L2 |
| --- | ---: | ---: | ---: |
| A0 | 0.00302684 | 0.00000000 | 0.00302684 |
| A1 | 0.00139001 | 0.00526688 | 0.00702829 |
| L1 | 0.00166567 | 0.00295345 | 0.00492095 |
| L2 | 0.00240769 | 0.00061777 | 0.00292001 |

## Recorded component costs

| Arm | Trainable adapter parameters | Head fit s/head | Adapter ms/image | Adapter + NN ms/image |
| --- | ---: | ---: | ---: | ---: |
| A0 | 0 | 0.000 | 0.101 | 37.692 |
| A1 | 24,992 | 40.062 | 1.629 | 39.533 |
| L1 | 24,992 | 20.432 | 1.354 | 38.914 |
| L2 | 24,992 | 20.155 | 1.352 | 38.910 |

- Head fit excludes feature extraction, diagnostics and checkpoint saving. Historical A1 includes the old four-forward/1,024-nominal-patch implementation; fresh arms compute the equivalent 512-active-center objective without zero-weight branches. Historical versus new fit times therefore do not isolate loss-function computational cost.
- Cached-feature adapter + NN timing excludes encoder extraction, query transfer, patch return, image-score aggregation and map processing. It is not measured end-to-end single-image latency. Mapping times are batch-amortized components. Shared feature extraction must not be repeatedly summed across arms.
- Training GPU peaks include the cached normal features and head/optimizer. Evaluation peaks also include the loaded encoder, reference bank and metric computation. They are not adapter-only or deployment-only memory. Missing historical measurements stay blank.

## Complete tables

- `metrics.csv`: all 1,260 cells; `paired_conditions.csv`: all within-cell comparisons.
- `category_seed_pairs.csv`, `category_paired_summary.csv`: every paired category/seed and category mean.
- `category_results.csv`, `seed_results.csv`, `condition_results.csv`: full grouped scores.
- `diagnostics.csv`, `diagnostic_pairs.csv`, `diagnostic_summary.csv`: common unweighted normal diagnostics.
- `costs.csv`: component timing and available memory measurements; `summary.json`: exact aggregates and gates.
