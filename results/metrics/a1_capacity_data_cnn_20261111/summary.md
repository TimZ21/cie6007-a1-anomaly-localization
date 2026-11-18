# Fixed A1 capacity, training-budget and spatial-CNN study

Complete matrix: **2,520 evaluations**, including 630 reused 16-shot controls and 1,890 new evaluations.

Paired differences are formed within category, seed and condition. Categories and conditions receive equal weight within each seed. Mean and sample SD are then reported across the three seeds.

All categories were evaluated in earlier studies. These groups are not newly unseen validation sets. Seeds reuse the same test images and are not independent datasets; a positive mean or a category count is not a significance test.

The primary metric is six-shift macro pixel AP. Scores below are percentages; paired changes are percentage points. The ± value is sample SD across the three paired seeds.

## all15

| Arm | Shift pixel AP | Clean pixel AP | Shift image AUROC | Clean image AUROC |
| --- | ---: | ---: | ---: | ---: |
| A0-16 | 57.147 ± 0.458 | 57.647 ± 0.441 | 98.141 ± 0.158 | 98.213 ± 0.138 |
| A0-32 | 57.547 ± 0.134 | 58.094 ± 0.091 | 98.405 ± 0.172 | 98.526 ± 0.183 |
| MLP32-16 | 58.488 ± 0.574 | 58.729 ± 0.566 | 98.381 ± 0.168 | 98.352 ± 0.139 |
| MLP64-16 | 58.443 ± 0.599 | 58.677 ± 0.588 | 98.373 ± 0.159 | 98.344 ± 0.135 |
| MLP32-32 | 58.881 ± 0.157 | 59.121 ± 0.117 | 98.675 ± 0.163 | 98.654 ± 0.177 |
| MLP64-32 | 58.939 ± 0.247 | 59.172 ± 0.213 | 98.673 ± 0.169 | 98.644 ± 0.191 |
| CNN1-32 | 58.957 ± 0.287 | 59.183 ± 0.246 | 98.661 ± 0.156 | 98.645 ± 0.165 |
| CNN3-32 | 58.852 ± 0.197 | 59.074 ± 0.164 | 98.639 ± 0.180 | 98.606 ± 0.185 |

| Prespecified comparison | Shift pixel AP Δ | Clean pixel AP Δ | Categories + / tie / − | Positive seed means |
| --- | ---: | ---: | ---: | ---: |
| MLP64-16_minus_MLP32-16 | -0.045 ± 0.031 | -0.051 ± 0.029 | 3 / 7 / 5 | 0 / 3 |
| MLP32-32_minus_MLP32-16 | +0.393 ± 0.522 | +0.393 ± 0.498 | 9 / 0 / 6 | 2 / 3 |
| MLP64-32_minus_MLP64-16 | +0.496 ± 0.509 | +0.494 ± 0.474 | 10 / 0 / 5 | 3 / 3 |
| MLP64-32_minus_MLP32-32 | +0.058 ± 0.092 | +0.050 ± 0.097 | 4 / 6 / 5 | 2 / 3 |
| MLP32-16_minus_A0-16 | +1.341 ± 0.249 | +1.082 ± 0.232 | 12 / 1 / 2 | 3 / 3 |
| MLP64-16_minus_A0-16 | +1.296 ± 0.252 | +1.030 ± 0.234 | 11 / 1 / 3 | 3 / 3 |
| MLP32-32_minus_A0-32 | +1.335 ± 0.104 | +1.027 ± 0.112 | 12 / 0 / 3 | 3 / 3 |
| MLP64-32_minus_A0-32 | +1.393 ± 0.181 | +1.077 ± 0.190 | 11 / 0 / 4 | 3 / 3 |
| CNN1-32_minus_A0-32 | +1.410 ± 0.215 | +1.089 ± 0.218 | 11 / 0 / 4 | 3 / 3 |
| CNN3-32_minus_A0-32 | +1.306 ± 0.153 | +0.980 ± 0.164 | 11 / 0 / 4 | 3 / 3 |
| CNN1-32_minus_MLP64-32 | +0.018 ± 0.039 | +0.012 ± 0.033 | 6 / 3 / 6 | 2 / 3 |
| CNN3-32_minus_CNN1-32 | -0.104 ± 0.092 | -0.109 ± 0.091 | 2 / 7 / 6 | 1 / 3 |
| A0-32_minus_A0-16 | +0.400 ± 0.453 | +0.447 ± 0.451 | 9 / 1 / 5 | 2 / 3 |
| width_by_data_interaction | +0.103 ± 0.063 | +0.102 ± 0.068 | 6 / 9 / 0 | 3 / 3 |

Mean paired six-shift pixel AP difference over three seeds; > +0.1 pp improved, < -0.1 pp degraded, otherwise tied.

## original3_development

| Arm | Shift pixel AP | Clean pixel AP | Shift image AUROC | Clean image AUROC |
| --- | ---: | ---: | ---: | ---: |
| A0-16 | 68.149 ± 0.083 | 68.303 ± 0.089 | 98.343 ± 0.215 | 98.293 ± 0.226 |
| A0-32 | 67.914 ± 0.270 | 68.057 ± 0.274 | 98.652 ± 0.127 | 98.601 ± 0.125 |
| MLP32-16 | 70.123 ± 0.363 | 70.258 ± 0.381 | 98.441 ± 0.148 | 98.372 ± 0.173 |
| MLP64-16 | 70.188 ± 0.233 | 70.326 ± 0.254 | 98.424 ± 0.119 | 98.349 ± 0.128 |
| MLP32-32 | 69.846 ± 0.577 | 69.958 ± 0.580 | 98.645 ± 0.150 | 98.540 ± 0.159 |
| MLP64-32 | 69.989 ± 0.496 | 70.099 ± 0.493 | 98.646 ± 0.154 | 98.549 ± 0.157 |
| CNN1-32 | 69.857 ± 0.535 | 69.969 ± 0.530 | 98.587 ± 0.157 | 98.519 ± 0.162 |
| CNN3-32 | 69.623 ± 0.588 | 69.695 ± 0.593 | 98.557 ± 0.152 | 98.442 ± 0.136 |

| Prespecified comparison | Shift pixel AP Δ | Clean pixel AP Δ | Categories + / tie / − | Positive seed means |
| --- | ---: | ---: | ---: | ---: |
| MLP64-16_minus_MLP32-16 | +0.065 ± 0.215 | +0.067 ± 0.217 | 1 / 2 / 0 | 1 / 3 |
| MLP32-32_minus_MLP32-16 | -0.277 ± 0.452 | -0.300 ± 0.452 | 1 / 0 / 2 | 1 / 3 |
| MLP64-32_minus_MLP64-16 | -0.199 ± 0.320 | -0.226 ± 0.307 | 1 / 0 / 2 | 1 / 3 |
| MLP64-32_minus_MLP32-32 | +0.143 ± 0.150 | +0.141 ± 0.151 | 2 / 0 / 1 | 2 / 3 |
| MLP32-16_minus_A0-16 | +1.974 ± 0.344 | +1.955 ± 0.335 | 3 / 0 / 0 | 3 / 3 |
| MLP64-16_minus_A0-16 | +2.040 ± 0.173 | +2.022 ± 0.173 | 3 / 0 / 0 | 3 / 3 |
| MLP32-32_minus_A0-32 | +1.931 ± 0.328 | +1.901 ± 0.323 | 3 / 0 / 0 | 3 / 3 |
| MLP64-32_minus_A0-32 | +2.075 ± 0.227 | +2.042 ± 0.219 | 3 / 0 / 0 | 3 / 3 |
| CNN1-32_minus_A0-32 | +1.943 ± 0.265 | +1.912 ± 0.256 | 3 / 0 / 0 | 3 / 3 |
| CNN3-32_minus_A0-32 | +1.709 ± 0.345 | +1.638 ± 0.351 | 3 / 0 / 0 | 3 / 3 |
| CNN1-32_minus_MLP64-32 | -0.132 ± 0.052 | -0.130 ± 0.048 | 1 / 0 / 2 | 0 / 3 |
| CNN3-32_minus_CNN1-32 | -0.234 ± 0.188 | -0.274 ± 0.194 | 0 / 1 / 2 | 0 / 3 |
| A0-32_minus_A0-16 | -0.234 ± 0.187 | -0.246 ± 0.188 | 1 / 0 / 2 | 0 / 3 |
| width_by_data_interaction | +0.078 ± 0.134 | +0.074 ± 0.146 | 2 / 1 / 0 | 2 / 3 |

Mean paired six-shift pixel AP difference over three seeds; > +0.1 pp improved, < -0.1 pp degraded, otherwise tied.

## prior12_extension

| Arm | Shift pixel AP | Clean pixel AP | Shift image AUROC | Clean image AUROC |
| --- | ---: | ---: | ---: | ---: |
| A0-16 | 54.396 ± 0.559 | 54.983 ± 0.533 | 98.090 ± 0.189 | 98.193 ± 0.180 |
| A0-32 | 54.955 ± 0.119 | 55.604 ± 0.082 | 98.343 ± 0.208 | 98.507 ± 0.213 |
| MLP32-16 | 55.579 ± 0.626 | 55.846 ± 0.613 | 98.366 ± 0.203 | 98.347 ± 0.175 |
| MLP64-16 | 55.506 ± 0.701 | 55.765 ± 0.682 | 98.360 ± 0.192 | 98.343 ± 0.172 |
| MLP32-32 | 56.140 ± 0.061 | 56.412 ± 0.009 | 98.683 ± 0.208 | 98.683 ± 0.227 |
| MLP64-32 | 56.177 ± 0.187 | 56.440 ± 0.152 | 98.680 ± 0.211 | 98.668 ± 0.242 |
| CNN1-32 | 56.232 ± 0.228 | 56.487 ± 0.186 | 98.679 ± 0.202 | 98.676 ± 0.221 |
| CNN3-32 | 56.160 ± 0.151 | 56.419 ± 0.155 | 98.660 ± 0.222 | 98.647 ± 0.236 |

| Prespecified comparison | Shift pixel AP Δ | Clean pixel AP Δ | Categories + / tie / − | Positive seed means |
| --- | ---: | ---: | ---: | ---: |
| MLP64-16_minus_MLP32-16 | -0.073 ± 0.075 | -0.081 ± 0.070 | 2 / 5 / 5 | 1 / 3 |
| MLP32-32_minus_MLP32-16 | +0.561 ± 0.629 | +0.566 ± 0.605 | 8 / 0 / 4 | 2 / 3 |
| MLP64-32_minus_MLP64-16 | +0.670 ± 0.613 | +0.674 ± 0.571 | 9 / 0 / 3 | 3 / 3 |
| MLP64-32_minus_MLP32-32 | +0.037 ± 0.142 | +0.028 ± 0.149 | 2 / 6 / 4 | 2 / 3 |
| MLP32-16_minus_A0-16 | +1.183 ± 0.249 | +0.863 ± 0.228 | 9 / 1 / 2 | 3 / 3 |
| MLP64-16_minus_A0-16 | +1.110 ± 0.298 | +0.783 ± 0.277 | 8 / 1 / 3 | 3 / 3 |
| MLP32-32_minus_A0-32 | +1.186 ± 0.076 | +0.808 ± 0.091 | 9 / 0 / 3 | 3 / 3 |
| MLP64-32_minus_A0-32 | +1.222 ± 0.183 | +0.836 ± 0.198 | 8 / 0 / 4 | 3 / 3 |
| CNN1-32_minus_A0-32 | +1.277 ± 0.220 | +0.883 ± 0.230 | 8 / 0 / 4 | 3 / 3 |
| CNN3-32_minus_A0-32 | +1.205 ± 0.212 | +0.816 ± 0.232 | 8 / 0 / 4 | 3 / 3 |
| CNN1-32_minus_MLP64-32 | +0.055 ± 0.042 | +0.047 ± 0.035 | 5 / 3 / 4 | 3 / 3 |
| CNN3-32_minus_CNN1-32 | -0.072 ± 0.130 | -0.068 ± 0.133 | 2 / 6 / 4 | 1 / 3 |
| A0-32_minus_A0-16 | +0.558 ± 0.591 | +0.621 ± 0.587 | 8 / 1 / 3 | 2 / 3 |
| width_by_data_interaction | +0.110 ± 0.106 | +0.109 ± 0.115 | 4 / 8 / 0 | 2 / 3 |

Mean paired six-shift pixel AP difference over three seeds; > +0.1 pp improved, < -0.1 pp degraded, otherwise tied.

## Interpretation and costs

- Width contrasts hold the shot budget fixed; data contrasts hold MLP width fixed.
- Moving to 32 shots expands both the adapter fitting set and the normal reference bank. The A0-32 − A0-16 contrast measures the reference-bank change without an adapter; same-budget adapter contrasts measure the additional benefit of the adapter.
- A positive width-by-data interaction means the width advantage is larger at 32 shots. It is a paired descriptive contrast, not evidence of statistical significance.
- CNN1 adds local spatial processing and parameters; CNN3 adds both receptive-field depth and parameters. These contrasts do not isolate depth at equal parameter count.
- Timing is reported in costs.csv. Adapter + nearest-neighbor scoring excludes the shared frozen-encoder extraction and map postprocessing; mapping has a separate column. Historical reused timings are labelled and may reflect different machine load. Unavailable per-arm peak-memory observations remain blank.
- Per-category, per-seed and per-condition tables retain all settings and regressions. No new loss-function experiment is part of this study.

| Arm | Parameters | Recorded head-training total (s) | Adapter + NN macro ms/image |
| --- | ---: | ---: | ---: |
| A0-16 | 0 | 0.00 | 37.692 |
| A0-32 | 0 | 0.00 | 74.981 |
| MLP32-16 | 24,992 | 1802.78 | 39.533 |
| MLP64-16 | 49,600 | 1218.69 | 38.941 |
| MLP32-32 | 24,992 | 1223.27 | 76.104 |
| MLP64-32 | 49,600 | 1215.95 | 75.994 |
| CNN1-32 | 86,528 | 1753.05 | 77.217 |
| CNN3-32 | 160,384 | 2472.78 | 78.275 |

## Detailed files

- `metrics.csv`: all 2,520 metric cells.
- `paired_conditions.csv`: every prespecified within-cell contrast.
- `category_seed_pairs.csv`: paired clean, shift and all-condition changes for each category and seed.
- `category_paired_summary.csv`: category-level paired means, seed SDs and the fixed practical direction labels.
- `category_results.csv`, `seed_results.csv`, `condition_results.csv`: complete grouped scores.
- `costs.csv`: parameter counts, recorded times and available peak-memory measurements.
- `summary.json`: numeric aggregates and the manifest-level analysis checks.
