"""Small residual adapter and aligned synthetic-defect quadruplets.

Research candidate, not an established algorithm or a claim of priority.
The frozen encoder, reference selection, nearest-neighbor rule and metrics
remain in the existing AnomalyDINO protocol. No real defect labels enter here.
"""
import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.features import photometric


class DefectAdapter(nn.Module):
    def __init__(self, dimension=384, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z):
        return F.normalize(z + self.up(F.gelu(self.down(z))), dim=-1)


def image_seed(path, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{path}'.encode()).digest()[:8], 'little')


def quadruplets(rgb, path, seed, conditions):
    """Float RGB [3,448,448]; same local defect and T in each four-view tuple.

    Two defect families x four training photometric transforms = eight tuples.
    The donor is from the SAME normal image. Interior pixels are inset by a
    full patch before pooling; no true anomaly/mask or external donor is used.
    """
    assert rgb.shape == (3, 448, 448) and rgb.device.type == 'cpu'
    rng = np.random.default_rng(image_seed(path, seed))
    for kind in ['cutpaste', 'local_color']:
        for condition in conditions:
            height, width = (int(x) for x in rng.integers(84, 141, size=2))
            top = int(rng.integers(14, 448 - height - 14 + 1))
            left = int(rng.integers(14, 448 - width - 14 + 1))
            changed = rgb.clone()
            meta = {'path': path, 'kind': kind, 'condition': condition,
                    'rectangle_tlhw': [top, left, height, width]}
            if kind == 'cutpaste':
                sy = int(rng.integers(0, 448 - height + 1))
                sx = int(rng.integers(0, 448 - width + 1))
                donor = rgb[:, sy:sy + height, sx:sx + width].flip(-1)
                changed[:, top:top + height, left:left + width] = donor
                meta['donor_top_left'] = [sy, sx]
            else:
                offset = torch.from_numpy(rng.choice([-1., 1.], size=3).astype(np.float32)) * .18
                changed[:, top:top + height, left:left + width] = (
                    changed[:, top:top + height, left:left + width] + offset[:, None, None]).clamp(0, 1)
                meta['rgb_offset'] = offset.tolist()
            interior = torch.zeros(1, 1, 448, 448)
            interior[:, :, top + 14:top + height - 14, left + 14:left + width - 14] = 1
            mask = (F.avg_pool2d(interior, 14, 14).flatten() >= .8)
            assert mask.any()
            shifted = photometric(rgb, condition['kind'], condition['factor'])
            shifted_changed = photometric(changed, condition['kind'], condition['factor'])
            yield (rgb, shifted, changed, shifted_changed), mask, meta


def loss_components(adapter, batch, normal_count):
    """Uniform normal patches followed by synthetic-interior patches.

    All trainable variants compute the same four forwards and loss terms.
    Only the coefficients selecting clean vs cross-light defect preservation
    differ. Teacher differences are frozen targets (not student differences).
    """
    z, zt, za, zta = batch
    h, ht, ha, hta = [adapter(x) for x in batch]
    delta = (za[normal_count:] - z[normal_count:]).detach()
    return {
        'illumination': (ht[:normal_count] - h[:normal_count]).square().sum(-1).mean(),
        'anchor': (h[:normal_count] - z[:normal_count]).square().sum(-1).mean(),
        'clean_defect': (ha[normal_count:] - h[normal_count:] - delta).square().sum(-1).mean(),
        'cross_defect': (hta[normal_count:] - ht[normal_count:] - delta).square().sum(-1).mean(),
    }


def objective(parts, method, anchor_weight=.1, defect_weight=1.):
    if method not in ('A1', 'A2', 'A3'):
        raise ValueError(method)
    # Retain identical computation graphs/batches even for the zero coefficients.
    return (parts['illumination'] + anchor_weight * parts['anchor']
            + defect_weight * (int(method == 'A2') * parts['clean_defect']
                               + int(method == 'A3') * parts['cross_defect']))


def batch_indices(rng, size, interior_indices, batch_size):
    assert batch_size % 2 == 0 and len(interior_indices)
    return np.concatenate([rng.integers(size, size=batch_size // 2),
                           rng.choice(interior_indices, size=batch_size // 2)])
