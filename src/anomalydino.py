"""Protocol adapter for AnomalyDINO (WACV 2025), Apache-2.0.

Based on dammsi/AnomalyDINO b9d1c2648e3a5247437d4d953d907a8f3d994457:
backbones.py, detection.py, utils.py, post_eval.py. See THIRD_PARTY.md.
Preserves official 448px input, agnostic rotations, normalized 1NN,
top-1% image score and native-resolution smoothing. Uses paired manifests,
our lighting conditions, a common 224px metric grid, and exact Torch GPU
search validated against CPU FAISS. These adaptations are explicitly logged.
"""
import importlib.util
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torchvision.transforms import functional as TF, InterpolationMode
from src.features import photometric, MEAN, STD

DINO_SHA='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
AD_SHA='b9d1c2648e3a5247437d4d953d907a8f3d994457'


def load(model_name='dinov2_vits14'):
    """Load a pinned local S/14 (default) or B/14 checkpoint, without downloads."""
    if model_name not in ('dinov2_vits14', 'dinov2_vitb14'):
        raise ValueError(f'Unsupported backbone: {model_name}')
    sys.path.insert(0,str(Path('cache/upstream/dinov2').resolve()))
    from dinov2.hub import backbones
    model=getattr(backbones,model_name)(pretrained=False)
    model.load_state_dict(torch.load(f'cache/torch/hub/checkpoints/{model_name}_pretrain.pth',map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False).cuda()


def prepare(image,condition):
    im=TF.resize(Image.fromarray(image),448,InterpolationMode.BICUBIC,antialias=True)
    rgb=photometric(TF.to_tensor(im),condition['kind'],condition['factor'])
    x=TF.normalize(rgb,MEAN,STD)
    return x[:,:x.shape[1]//14*14,:x.shape[2]//14*14]


@torch.inference_mode()
def features(model,image,condition):
    x=prepare(image,condition)
    assert x.shape==(3,448,448), 'This fixed three-category protocol expects square images'
    return model.get_intermediate_layers(x.unsqueeze(0).cuda())[0].squeeze(0).contiguous()


def rotations(image):
    center=tuple(np.array(image.shape[1::-1])/2)
    return [cv2.warpAffine(image,cv2.getRotationMatrix2D(center,angle,1),image.shape[1::-1],
            flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_DEFAULT) for angle in range(0,360,45)]


@torch.inference_mode()
def distances(queries,bank,chunk=128):
    # Both inputs are FAISS-normalized float32; same squared L2/2 formula.
    bn=bank.square().sum(1)
    return torch.cat([((q.square().sum(1)[:,None]+bn[None,:]-2*(q@bank.T)).clamp_min_(0).min(1).values*.5)
                      for q in queries.split(chunk)])


def image_score(d):
    return float(np.mean(np.sort(np.asarray(d,dtype=float).flatten())[::-1][:max(1,int(d.size*.01))]))


def map224(d,shape):
    native=cv2.resize(np.asarray(d,dtype=float).reshape(32,32),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    native=gaussian_filter(native,sigma=4)
    return cv2.resize(native,(224,224),interpolation=cv2.INTER_LINEAR)


def upstream_module(name):
    spec=importlib.util.spec_from_file_location('official_'+name,Path('cache/upstream/AnomalyDINO/src')/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
