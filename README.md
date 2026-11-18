# A1: a small residual adapter for illumination-robust anomaly localization

Public reproduction materials for the CIE6007 course project. The final method
is **A1 (MLP32, 16-shot)**: a frozen DINOv2-S/14 encoder plus a 24,992-parameter
residual adapter trained on normal images. It is evaluated against our fixed
AnomalyDINO-based A0 implementation on all 15 MVTec AD categories. This is a
controlled robustness study, not a claim of a new encoder or state of the art.

The original development repository, frozen results and report are preserved
separately. This repository contains the necessary scientific implementation,
fixed configurations and image splits, small audited result tables, and portable
reproduction commands. Personal submission documents, data, weights, large
feature caches and checkpoints are not redistributed.

## Method and protocol

For a normalized 384-dimensional patch feature `z`,

```
h(z) = normalize(z + W_up GELU(W_down z + b_down) + b_up)
L_A1 = mean_i ||h(T(z_i)) - h(z_i)||² + 0.1 mean_i ||h(z_i) - z_i||²
```

`T(z)` denotes the feature extracted from the photometrically transformed
image, not a transformation directly applied to a feature. The output projection
starts at zero. The first term aligns normal features across illumination; the
anchor limits movement from the frozen clean descriptor. It does not guarantee
preservation of every defect, and cable is a relevant failure case.

- Each category and seed has its own adapter and reference bank. Seeds: 0, 1, 2.
- 16-shot: 12 normal images for training, 4 for diagnostics; all 16 for the bank.
  32-shot adds 12 training and 4 diagnostic images while retaining the original
  16. Diagnostics never select hyperparameters or checkpoints.
- DINOv2-S/14, last patch-token layer, 384 dimensions, 448-pixel short-edge input.
  Feature grids use their actual dimensions. The audited images yield 32×32.
- AdamW, learning rate 0.001, weight decay 0.0001, 1,000 fixed steps, 512 active
  normal centers per step; feature-dimension squared errors are summed before
  averaging over centers. Historical inactive RNG draws are retained.
- Training light views: brightness 0.85/1.15 and contrast 0.85/1.15.
- Test: clean, brightness 0.70/1.30, contrast 0.70/1.30, gamma 0.70/1.40.
- Eight reference rotations; exact 1-nearest-neighbor normalized feature search;
  image score is the mean of the largest 1% of patch distances. Localization
  resizes to native resolution, applies Gaussian smoothing with σ=4 native
  pixels, then evaluates on a 224×224 grid. Ground-truth masks use nearest-neighbor
  resizing. All official test images are evaluated.
- No real anomaly labels or test images enter training or checkpoint selection.

`src/defect_adapter.py`, `src/capacity_adapter.py` and `src/loss_study.py` contain
the trainable methods. `src/anomalydino.py` contains the inherited scoring and
photometric protocol. These files are byte-identical to the research versions.
Three original scripts cited by the report (`run_defect_adapter.py`,
`analyze_mvtec_extension.py`, `analyze_defect_adapter.py`) are also retained as
source references. They depend on unbundled historical helpers, caches or run
records and are **not** the portable entrypoints. Use `reproduce.py` and
`analyze_release.py` below.

Legacy unused helper functions remain in `features.py` and `banks.py` to preserve
that dependency closure; they are not alternative encoders used by this release.

## What the results support

A1 provides a modest average robustness benefit, with meaningful category
variation. Wider MLPs, more normal training/reference images, spatial CNN blocks,
and the two loss variants do not establish a consistently better replacement.
The MLP768 gain seen in the initial three-category study was driven largely by
carpet; its full-15-category result supersedes that early comparison. The final
choice remains the compact A1. Use the complete tables, including deteriorations,
not just favorable examples. Costs are measured on an RTX 4060 laptop; shared
hardware, caching and measurement scope affect comparisons.

Read [results/README.md](results/README.md) for the eight study directories and
[PUBLICATION.md](PUBLICATION.md) for the curated-artifact provenance. Original
three-category development results and the other 12 categories are separated;
those 12 had already been evaluated before later ablations and are not a newly
unseen validation set for those ablations. The three seeds share the same test
set, so sample standard deviations are descriptive and do not prove significance.

## Reanalyze the bundled tables (no GPU or dataset)

Python 3.9 or later is sufficient for this command; it uses the standard library.
Run from the repository root:

```powershell
python scripts/analyze_release.py --out output/reanalysis
```

It verifies all 5,517 archived cells across eight studies, including reused
controls; this count is not the number of independent new experiments. It writes
method summaries, every requested cell-level paired difference, aggregate paired
summaries, and category classifications. Differences are formed within the same
category/seed/condition, then averaged with equal category and condition weights.
Seed means and sample SDs are reported in percentage points. A category improves
above +0.1 pp, degrades below −0.1 pp, and otherwise ties, using mean shifted Pixel
AP. Use a new output path on a repeat invocation.

The archived CSVs preserve the original full precision. Published summary and
cost/diagnostic tables remain under `results/metrics/`; their timing scopes are
recorded there. Hashes of private large artifacts are provenance identities, not
promises that those artifacts are included in this public release.

## Windows GPU environment

The recorded experiments used Windows, Python **3.11.9**, an **RTX 4060 Laptop
GPU (8 GB)** and 16 GB system RAM, with PyTorch **2.13.0+cu126**. Install a
compatible NVIDIA driver. A separate CUDA toolkit is not needed for the supplied
PyTorch wheels. Use a new local clone and run commands from its root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/windows-core.lock.txt
.\.venv\Scripts\python.exe -m pip install -r requirements/torch-cu126.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements/windows-e2.lock.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The lock files record the study environment. Different libraries/hardware can
introduce floating-point differences; do not infer byte-identical checkpoints
across platforms. Exact published checkpoint hashes are in
`results/checkpoint_identities.json`.

## Data, weights and preflight

1. Obtain the **original MVTec AD** dataset from its
   [official dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad).
   Follow its license. Extract the category directories into `data/mvtec_ad/`
   (for example `data/mvtec_ad/bottle/train/good/000.png`). The 15-category image
   and mask inventory is `assets/mvtec_extension_files.json`. Do not use AD 2.
2. Download the official DINOv2 source at the frozen Git revision and the official
   S/14 pretrained weight, then verify the recorded local SHA-256 identity:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_encoder.py
.\.venv\Scripts\python.exe scripts/reproduce.py --check-schedules
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The downloader uses official Facebook Research GitHub/Meta download endpoints
and preserves the upstream license. It installs local files under `cache/`.
The SHA-256 is our downloaded-file identity, not a publisher-provided checksum.
Sampling verification recomputes all 90 category/seed/shot schedules against the
archived hashes in `assets/center_schedules.json`. Unit tests use synthetic
features to check initialization, losses/gradients, CNN neighborhoods and gradient
accumulation; they do not perform dataset experiments.

The exact ordered image roles are recorded in
`results/metrics/a1_capacity_data_cnn_20261111/splits.json`. Do not resample them.
Each run verifies all selected-category dataset bytes, official weight identity,
and pinned encoder source before training. Several GB of free disk space are
needed for normal-feature caches, maps and checkpoints; keep large outputs local.

## Fresh reproduction

`reproduce.py` is a **new portable orchestration wrapper** around the unchanged
scientific functions. It replaces historical private-Git and cache gates. The
bundled results were generated by the original research runners, not by this new
wrapper. Packaging checks are not a new full rerun and do not establish identical
fresh metrics; compare a new run with the bundled per-cell table before claiming
numerical reproduction.

The wrapper validates fixed splits and sampling, trains **all requested final
1,000-step checkpoints first**, then evaluates. Each command requires a new
output directory. A failure preserves logs and artifacts; use a new directory
for a retry. It never overwrites published results or performs tuning.

```powershell
# Final A0/A1 comparison: 15 categories, 3 seeds, 7 conditions (630 cells)
.\.venv\Scripts\python.exe scripts/reproduce.py --out runs/final_a0_a1

# Capacity/data/CNN arms: 2,520 cells; includes freshly recomputed controls
.\.venv\Scripts\python.exe scripts/reproduce.py --out runs/capacity --arms A0-16 A0-32 MLP32-16 MLP64-16 MLP32-32 MLP64-32 CNN1-32 CNN3-32

# Full-15 MLP768 comparison: 630 cells
.\.venv\Scripts\python.exe scripts/reproduce.py --out runs/wide --arms MLP32-16 MLP768-16

# Fixed MLP32/16-shot loss comparison: 1,260 cells
.\.venv\Scripts\python.exe scripts/reproduce.py --out runs/loss --arms A0-16 MLP32-16 L1 L2

# No-anchor ablation: the reported three-category scope, 189 cells
.\.venv\Scripts\python.exe scripts/reproduce.py --out runs/no_anchor --categories bottle cable carpet --arms A0-16 MLP32-16 A1-no-anchor

# Historical cable A0–A3 pilot: only its three recorded conditions, 36 cells
.\.venv\Scripts\python.exe scripts/reproduce.py --out runs/pilot --categories cable --arms A0-16 MLP32-16 A2 A3
```

`--data D:\datasets\mvtec_ad` changes only the dataset location. For a runtime
smoke test, explicitly use `--categories bottle --seeds 0`; it is a subset check,
not the reported full experiment. `--chunk 128` reduces the active-center gradient
accumulation chunk if needed, retaining the 512-center normalization and sampling.
A2/A3 retain their original full synthetic batch; do not interpret that option
as changing their batch. No public command adds an encoder, width, loss or tuning
sweep beyond the reported settings.

Fresh runs save configurations, source/weight identities, normal diagnostic
measurements, final-step adapter checkpoints and sampled training-loss logs, per-image scores,
patch-score files, costs and full metrics under the chosen run directory. The
score timing excludes encoder extraction; the separate evaluation time includes
encoder extraction and localization mapping. These release timings have a
different orchestration scope from the original same-session microbenchmarks and
should not be substituted for them in the report. Original A1 is called
`MLP32-16` in fresh-run tables, and original A0 is `A0-16`.

## Attribution

See [THIRD_PARTY.md](THIRD_PARTY.md), the upstream license files in `licenses/`,
and [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). Dataset and model rights remain
with their respective owners. This repository does not redistribute them.
