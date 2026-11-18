"""Two fixed loss variants on the unchanged original A1 pointwise adapter.

Inputs are cached, normalized features from the frozen DINOv2 encoder. Each
row is one of the same 512 paired normal/light centers used by original A1.
No synthetic defect, extra training sample, or learned teacher is introduced.
"""

import torch

from src.defect_adapter import DefectAdapter


def make_adapter(seed, device='cuda'):
    """Preserve the original 384 -> 32 -> 384 architecture and initialization."""
    torch.manual_seed(5000 + seed)
    return DefectAdapter(dimension=384, bottleneck=32).to(device)


def loss_parts(head, z, zt, total_centers=None):
    """Return center-normalized loss terms for one complete batch or chunk.

    Sum the feature dimension, then divide by the full effective batch size.
    For gradient accumulation, every chunk receives ``total_centers=512``;
    callers sum its loss gradients before a single optimizer step. ``z`` and
    ``zt`` must contain matching clean/light centers in matching row order.
    Only the clean encoder feature is the fixed target: never detach the
    trainable clean output and use it as a replacement teacher.
    """
    if z.ndim != 2 or zt.shape != z.shape or len(z) == 0:
        raise ValueError('Expected nonempty, matching B-by-D clean/light features.')
    denominator = len(z) if total_centers is None else total_centers
    if not isinstance(denominator, int) or isinstance(denominator, bool) or denominator < len(z):
        raise ValueError('total_centers must be an integer at least as large as this chunk.')
    h, ht = head(z), head(zt)
    teacher = z.detach()

    def squared_distance(a, b):
        return (a - b).square().sum(-1).sum() / denominator

    return {
        'illumination': squared_distance(ht, h),
        'anchor': squared_distance(h, teacher),
        'fixed_teacher_alignment': squared_distance(ht, teacher),
    }


def total(parts, variant):
    """A1 control, doubled anchor L1, or fixed-clean-teacher alignment L2."""
    if variant == 'A1':
        return parts['illumination'] + .1 * parts['anchor']
    if variant == 'L1':
        return parts['illumination'] + .2 * parts['anchor']
    if variant == 'L2':
        return parts['fixed_teacher_alignment'] + .1 * parts['anchor']
    raise ValueError(f'Unknown loss-study variant: {variant}')
