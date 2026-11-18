"""Shared scoring with exact, tie-aware average precision on the GPU."""
import torch
from torch.nn import functional as F
from .banks import nearest_distance


@torch.inference_mode()
def pixel_average_precision(labels, scores):
    """Non-interpolated AP, grouping equal scores as sklearn does.

    Integer cumulative counts and float64 reduction avoid float32 summation
    error. Sorting and grouping change execution location, not the metric.
    """
    labels, scores = labels.flatten(), scores.flatten()
    assert len(labels) == len(scores) and torch.isfinite(scores).all()
    values, order = scores.sort(descending=True)
    tp = labels[order].to(torch.int64).cumsum(0)
    ends = torch.cat((values[:-1] != values[1:], values.new_ones(1, dtype=torch.bool)))
    indices = ends.nonzero().flatten()
    positives = tp[indices]
    total = positives[-1]
    if total == 0:
        return 0.0
    increments = torch.diff(positives, prepend=positives.new_zeros(1))
    precision = positives.double() / (indices + 1).double()
    return ((increments.double() * precision).sum() / total).item()


@torch.inference_mode()
def score_features(features, bank, batch_size=2, chunk_size=1024):
    """Preserve the pilot's per-batch distance and upsampling shapes."""
    image_scores, maps = [], []
    for batch in features.split(batch_size):
        distances = nearest_distance(batch.reshape(-1, 384), bank, chunk_size).reshape(-1, 28, 28)
        image_scores.append(distances.flatten(1).max(1).values)
        maps.append(F.interpolate(distances[:, None], size=(224, 224),
                                  mode='bilinear', align_corners=False)[:, 0])
    return torch.cat(image_scores), torch.cat(maps)
