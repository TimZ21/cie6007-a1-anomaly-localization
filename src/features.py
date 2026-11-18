"""Frozen, full-image ResNet-18 patch descriptors for the shared B1/B2/P protocol.

This is our small PatchCore-style adaptation, not the original PatchCore code.
Torchvision supplies the backbone implementation and ImageNet weights.
"""
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.transforms import functional as TF, InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def photometric(rgb, kind='clean', factor=1.0):
    """Float RGB transform; contrast pivots on image-wide Rec.709 luminance."""
    if kind == 'clean':
        result = rgb
    elif kind == 'brightness':
        result = rgb * factor
    elif kind == 'contrast':
        weights = rgb.new_tensor([0.2126, 0.7152, 0.0722])[:, None, None]
        pivot = (rgb * weights).sum(dim=0).mean()
        result = pivot + factor * (rgb - pivot)
    elif kind == 'gamma':
        result = rgb.pow(factor)
    else:
        raise ValueError(f'Unknown transform: {kind}')
    return result.clamp(0, 1)


def preprocess(image: Image.Image, kind='clean', factor=1.0) -> torch.Tensor:
    """Full RGB resize, then float photometric transform, then normalization."""
    image = image.convert('RGB')
    image = TF.resize(image, [224, 224], InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(photometric(TF.to_tensor(image), kind, factor), MEAN, STD)


def combine_features(layer2, layer3):
    """3x3 local mean, bilinear alignment, then channel concatenation.

    Zero padding contributes to the mean at boundaries; keep this convention
    identical across all compared methods. No descriptor L2 normalization.
    """
    layer2 = F.avg_pool2d(layer2, 3, stride=1, padding=1, count_include_pad=True)
    layer3 = F.avg_pool2d(layer3, 3, stride=1, padding=1, count_include_pad=True)
    layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode='bilinear', align_corners=False)
    return torch.cat([layer2, layer3], dim=1)


class FrozenResNet18(nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        backbone = resnet18(weights=None)
        # weights_only avoids loading arbitrary pickled Python objects.
        backbone.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        """Keep batch-normalization statistics fixed even if caller enables train."""
        return super().train(False)

    @torch.inference_mode()
    def forward(self, images):
        x = self.layer1(self.stem(images))
        layer2 = self.layer2(x)
        layer3 = self.layer3(layer2)
        return combine_features(layer2, layer3)


def flatten_patches(feature_map):
    """[B,C,H,W] -> [B,H*W,C], row-major patches (row, column)."""
    return feature_map.flatten(2).transpose(1, 2).contiguous()
