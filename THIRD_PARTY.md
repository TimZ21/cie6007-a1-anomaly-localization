# Third-party sources

- [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), commit `b9d1c2648e3a5247437d4d953d907a8f3d994457`, Apache-2.0. `src/anomalydino.py` adapts preprocessing, rotations, normalized nearest-neighbor distance, image aggregation and map reconstruction from its `src/backbones.py`, `src/detection.py`, `src/utils.py` and `src/post_eval.py`. Reference selections use our frozen paired manifests; scores use the project's 224-pixel metric grid. This is a matched protocol adaptation, not an unmodified official benchmark run.
- [DINOv2](https://github.com/facebookresearch/dinov2), Copyright Meta Platforms, Inc. and affiliates, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, Apache-2.0. The frozen ViT-S/14 encoder is loaded from this pinned source and the official Meta checkpoint. The weight URL and locally recorded SHA-256 identity are in `assets/dinov2_backbones.json`; this is not a publisher-provided checksum.

The corresponding license texts are preserved in `licenses/`. The scientific modules retain their source notices. PyTorch, torchvision, NumPy, SciPy, scikit-learn, Pillow, OpenCV and other dependencies retain their respective upstream licenses.

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) is credited to Bergmann et al. (2019). Obtain the data separately under its CC BY-NC-SA 4.0 terms. This release includes selection paths and numeric results, not dataset images, masks, pretrained weights or feature caches.
