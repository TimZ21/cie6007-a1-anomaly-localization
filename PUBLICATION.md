# Public snapshot

This repository contains the implementation, configurations, saved evaluation tables and reproducibility metadata for the report. The original development repository and all experiment outputs were retained. Creating this public snapshot did not retrain a model or alter any metric, checkpoint, split or scientific conclusion.

## Included records

The result index is [results/README.md](results/README.md). [results/record_manifest.json](results/record_manifest.json) records SHA-256 identities and sizes for the selected original records. Metric CSVs, paired tables, configurations, image subsets and diagnostic tables are copied byte-for-byte. Recorded negative results are included. Historical controls reused across studies remain the same observations, not independent repetitions.

The public copy differs from the original metadata only as follows:

- In the capacity study's `pairing_verification.json`, the original Windows working-directory prefix is removed from 630 provenance path strings. They now use repository-relative paths. Audit checks, numbers and recorded content hashes are unchanged; the manifest retains both original and published file hashes.
- `assets/dinov2_backbones.json` contains the unchanged DINOv2-S/14 identity entry. The unused DINOv2-B/14 entry is omitted.
- `assets/center_schedules.json` contains the exact 90 schedule records selected from the capacity study's preflight record. `results/checkpoint_identities.json` collects checkpoint hashes and selected identity/cost fields from the original training records, with source-record hashes. These are compact metadata extracts, not new experiments.

Internal console logs, training histories, host paths, worktree administration, credentials, course materials and personalized report files are not distributed. Public reproduction instructions use configurable local directories. Historical metadata can refer to `runs/`, `units/` or `training/` records that are not bundled; their hashes identify preserved original artifacts.

## External data and weights

MVTec AD images, pretrained weights, feature caches, raw patch-score arrays and checkpoint binaries are not included. Obtain the dataset and weights from their official sources and follow the README. The dataset manifests contain file names, image metadata and SHA-256 hashes only. Dataset and pretrained-model usage remain subject to their respective terms.

## Packaging verification

[PUBLICATION_CHECKS.json](PUBLICATION_CHECKS.json) records the release checks: all 90 sampling schedules match, all 5,517 archived cells pass reanalysis, and 26 CPU tests pass. One CUDA-only test was skipped. The archived experiments used the recorded Windows GPU environment; the portable wrapper has not been used for a new full dataset rerun.
