"""Capacity-only adapters and real-grid center sampling for the frozen A1 loss.

The original pointwise adapter is imported unchanged. CNN training uses crops
with a halo as wide as its number of spatial blocks. A validity mask is applied
after every spatial layer: padding a crop only once would let learned biases
create nonexistent pixels beyond the original image boundary.
"""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.defect_adapter import DefectAdapter


class SpatialAdapter(nn.Module):
    """Pointwise projections surrounding one or three spatial residual blocks."""

    def __init__(self, depth, dimension=384, hidden=64, prototype=None):
        super().__init__()
        if depth not in (1, 3):
            raise ValueError('This study fixes CNN depth to 1 or 3.')
        if prototype is None:
            prototype = DefectAdapter(dimension=dimension, bottleneck=hidden)
        if prototype.down.weight.shape != (hidden, dimension):
            raise ValueError('Projection dimensions do not match the prototype.')
        self.depth = depth
        self.down = nn.Conv2d(dimension, hidden, 1)
        self.blocks = nn.ModuleList(
            nn.Conv2d(hidden, hidden, 3, stride=1, padding=1)
            for _ in range(depth)
        )
        self.up = nn.Conv2d(hidden, dimension, 1)
        with torch.no_grad():
            self.down.weight.copy_(prototype.down.weight[:, :, None, None])
            self.down.bias.copy_(prototype.down.bias)
            self.up.weight.copy_(prototype.up.weight[:, :, None, None])
            self.up.bias.copy_(prototype.up.bias)
        for block in self.blocks:
            nn.init.zeros_(block.weight)
            nn.init.zeros_(block.bias)

    def forward(self, z, valid=None):
        """BCHW features, with an optional B1HW mask for cropped image edges."""
        if z.ndim != 4:
            raise ValueError('SpatialAdapter requires BCHW features.')
        if valid is not None:
            if valid.shape != (z.shape[0], 1, z.shape[2], z.shape[3]):
                raise ValueError('Validity mask must have shape B1HW.')
            valid = valid.to(device=z.device, dtype=z.dtype)
        u = F.gelu(self.down(z))
        if valid is not None:
            u = u * valid
        for block in self.blocks:
            u = u + block(F.gelu(u))
            if valid is not None:
                u = u * valid
        h = F.normalize(z + self.up(u), dim=1)
        return h if valid is None else h * valid


def make_head(arm, seed, device='cuda'):
    """Initialize each arm reproducibly; all width-64 projections are shared."""
    specs = {
        'MLP32-16': (32, 0), 'MLP32-32': (32, 0),
        'MLP64-16': (64, 0), 'MLP64-32': (64, 0),
        'CNN1-32': (64, 1), 'CNN3-32': (64, 3),
    }
    if arm not in specs:
        raise ValueError(f'Unknown trainable arm: {arm}')
    width, depth = specs[arm]
    torch.manual_seed(5000 + seed)
    prototype = DefectAdapter(bottleneck=width)
    head = prototype if depth == 0 else SpatialAdapter(depth, prototype=prototype)
    return head.to(device)


def transform_grid(head, z):
    """Transform one HWD grid, preserving its actual height, width and device.

    The caller places the head and features on the same device. Each rotated
    reference image must be passed separately, before flattening its grid.
    """
    if z.ndim != 3:
        raise ValueError('Expected a single HWD grid.')
    if head is None:
        return z
    if isinstance(head, SpatialAdapter):
        return head(z.permute(2, 0, 1).unsqueeze(0))[0].permute(1, 2, 0).contiguous()
    return head(z)


def _gather_uniform(grids, view_ids, rows, cols, radius):
    """Vectorized gather for a tensor of equally shaped HWD views."""
    height, width = grids.shape[1:3]
    if ((view_ids < 0) | (view_ids >= len(grids))).any():
        raise IndexError('View index is outside the supplied feature grids.')
    if ((rows < 0) | (rows >= height) | (cols < 0) | (cols >= width)).any():
        raise IndexError('A sampled center is outside its real feature grid.')
    offsets = torch.arange(-radius, radius + 1, device=grids.device)
    yy = rows[:, None, None] + offsets[None, :, None]
    xx = cols[:, None, None] + offsets[None, None, :]
    valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
    values = grids[view_ids[:, None, None], yy.clamp(0, height - 1), xx.clamp(0, width - 1)]
    values = values * valid[..., None]
    return values.permute(0, 3, 1, 2).contiguous(), valid[:, None]


def gather_neighborhoods(grids, view_ids, rows, cols, radius):
    """Gather actual same-view neighbors; never join unrelated random patches.

    ``grids`` is a VHWD tensor or a sequence of HWD tensors (allowing different
    heights and widths). Indices are one-dimensional and have equal length.
    The returned BCHW crops and B1HW validity mask remain on the grids' device.
    Radius zero also provides an efficient pointwise training batch.
    """
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    if isinstance(grids, torch.Tensor):
        if grids.ndim != 4 or len(grids) == 0:
            raise ValueError('Tensor grids must have nonempty VHWD shape.')
        device = grids.device
    elif isinstance(grids, Sequence) and len(grids):
        device = grids[0].device
        if any(z.ndim != 3 or z.device != device or z.shape[-1] != grids[0].shape[-1]
               or z.dtype != grids[0].dtype for z in grids):
            raise ValueError('Sequence grids must be HWD with a common device, dtype and dimension.')
    else:
        raise ValueError('Supply a nonempty tensor or sequence of feature grids.')
    view_ids, rows, cols = [torch.as_tensor(x, dtype=torch.long, device=device)
                            for x in (view_ids, rows, cols)]
    if any(x.ndim != 1 for x in (view_ids, rows, cols)) or not (len(view_ids) == len(rows) == len(cols)):
        raise ValueError('View and center indices must be equal-length vectors.')
    if isinstance(grids, torch.Tensor):
        return _gather_uniform(grids, view_ids, rows, cols, radius)
    if ((view_ids < 0) | (view_ids >= len(grids))).any():
        raise IndexError('View index is outside the supplied feature grids.')
    size = 2 * radius + 1
    patches = grids[0].new_zeros((len(view_ids), grids[0].shape[-1], size, size))
    valid = torch.zeros((len(view_ids), 1, size, size), dtype=torch.bool, device=device)
    for view in view_ids.unique().tolist():
        selected = (view_ids == view).nonzero(as_tuple=True)[0]
        p, m = _gather_uniform(grids[view][None], torch.zeros_like(selected),
                               rows[selected], cols[selected], radius)
        patches[selected] = p
        valid[selected] = m
    return patches, valid


def center_features(head, patches, valid=None):
    """Adapt cropped BCHW features and return only their B-by-D centers.

    The center loss must be reduced over the complete effective batch, even
    when callers split these inputs into several gradient-accumulation chunks.
    """
    if patches.ndim == 2 and not isinstance(head, SpatialAdapter):
        return patches if head is None else head(patches)
    if patches.ndim != 4 or patches.shape[-2] % 2 != 1 or patches.shape[-1] % 2 != 1:
        raise ValueError('Expected BCHW crops with odd spatial dimensions.')
    row, col = patches.shape[-2] // 2, patches.shape[-1] // 2
    if isinstance(head, SpatialAdapter):
        if min(row, col) < head.depth:
            raise ValueError('Crop does not contain the complete CNN receptive field.')
        if valid is None:
            raise ValueError('Cropped CNN features require a real-grid validity mask.')
        if not bool(valid[:, 0, row, col].all()):
            raise ValueError('Each loss center must be a real feature-grid position.')
        return head(patches, valid)[:, :, row, col]
    z = patches[:, :, row, col]
    return z if head is None else head(z)
