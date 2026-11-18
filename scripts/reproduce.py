"""Portable release orchestration for the fixed reported protocol.

Scientific adapter, augmentation, sampling geometry, nearest-neighbor and AP
functions are imported unchanged. This replaces private-history/cache gates;
it does not replace or claim to have rerun the archived experiments.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_capacity_study import rectangle_metadata, interior_indices, sha256, save

ARMS = ('A0-16', 'A0-32', 'MLP32-16', 'MLP64-16', 'MLP32-32', 'MLP64-32',
        'CNN1-32', 'CNN3-32', 'MLP768-16', 'A1-no-anchor', 'L1', 'L2', 'A2', 'A3')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def protocol():
    return read(ROOT/'configs/a1_capacity_data_cnn.json')


def schedules(sub, cfg, grid=(32, 32), full=False):
    """Preserve active centers and the inactive historical RNG draws exactly."""
    import numpy as np
    records = [r for p in sub['fit_paths'] for r in rectangle_metadata(
        p, sub['seed'], cfg['training_conditions'], tuple(x*14 for x in grid))]
    interior = interior_indices(records, grid)
    rng = np.random.default_rng(2000+sub['seed'])
    active, legacy = [], []
    for _ in range(1000):
        ids = rng.integers(len(records)*grid[0]*grid[1], size=512)
        inactive = rng.choice(interior, size=512)
        active.append(ids); legacy.append(np.concatenate([ids, inactive]))
    active, legacy = np.asarray(active), np.asarray(legacy)
    hashes = {'active_schedule_sha256': hashlib.sha256(active.astype('<i8').tobytes()).hexdigest(),
              'legacy_full_schedule_sha256': hashlib.sha256(legacy.astype('<i8').tobytes()).hexdigest()}
    expected = next(r for r in read(ROOT/'assets/center_schedules.json')
                    if (r['category'], r['seed'], r['shots']) == (sub['category'], sub['seed'], sub['shots']))
    assert list(grid) == expected['grid_hw']
    for k, value in hashes.items():
        assert value == expected[k], (sub['category'], sub['seed'], sub['shots'], k)
    return (legacy if full else active), hashes


def verify_schedules():
    cfg = protocol()
    subs = read(ROOT/'results/metrics/a1_capacity_data_cnn_20261111/splits.json')['subsets']
    for sub in subs:
        schedules(sub, cfg)
    print(f'Verified {len(subs)} original active and historical full schedules.')


def verify_data(data, categories):
    inventory = read(ROOT/'assets/mvtec_extension_files.json')
    rows = [r for r in inventory if r['path'].split('/')[0] in categories]
    expected = {r['path'] for r in rows}
    actual = {p.relative_to(data).as_posix() for c in categories for p in (data/c).rglob('*.png')}
    assert actual == expected, {'missing': sorted(expected-actual), 'extra': sorted(actual-expected)}
    for r in rows:
        p = data/r['path']
        assert p.stat().st_size == r['bytes'] and sha256(p) == r['sha256'], p
    return {r['path']: r for r in rows}


def dependencies():
    global np, torch, cv2, faiss, ad, DefectAdapter, SpatialAdapter
    global make_head, transform_grid, gather_neighborhoods, center_features, loss_parts
    import numpy as np
    import torch
    import cv2
    import faiss
    from src import anomalydino as ad
    from src.defect_adapter import DefectAdapter
    from src.capacity_adapter import (SpatialAdapter, make_head, transform_grid,
                                      gather_neighborhoods, center_features)
    from src.loss_study import loss_parts
    assert sys.prefix != sys.base_prefix, 'Use the documented Python venv.'
    assert torch.cuda.is_available(), 'A CUDA GPU is required for the fixed experiment.'
    torch.set_num_threads(4); faiss.omp_set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0); np.random.seed(0)


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def normalize(z):
    values = np.ascontiguousarray(z.detach().cpu().numpy(), dtype=np.float32)
    faiss.normalize_L2(values.reshape(-1, 384))
    return torch.from_numpy(values)


def encode(model, image, condition):
    with torch.inference_mode():
        x = ad.prepare(image, condition)
        h, w = x.shape[-2]//14, x.shape[-1]//14
        z = model.get_intermediate_layers(x[None].cuda())[0][0].contiguous()
        assert z.shape == (h*w, 384)
        return normalize(z).reshape(h, w, 384)


def rgb(path):
    im = cv2.imread(str(path))
    assert im is not None, path
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def head_for(arm, seed):
    if arm.startswith('A0'):
        return None
    if arm in ('MLP32-16', 'MLP32-32', 'MLP64-16', 'MLP64-32', 'CNN1-32', 'CNN3-32'):
        return make_head(arm, seed)
    torch.manual_seed(5000+seed)
    return DefectAdapter(bottleneck=768 if arm == 'MLP768-16' else 32).cuda()


def normal_grids(model, names, data, cfg, out):
    values = []
    for name in names:
        dest = out/'normal_cache'/Path(name).with_suffix('.npy')
        if not dest.exists():
            image = rgb(data/name)
            z = torch.stack([encode(model, image, c) for c in [cfg['conditions'][0], *cfg['training_conditions']]])
            dest.parent.mkdir(parents=True, exist_ok=True); np.save(dest, z.numpy())
        values.append(np.load(dest, allow_pickle=False))
    assert len({v.shape[1:3] for v in values}) == 1
    return torch.from_numpy(np.concatenate(values)).cuda()


def backward_centers(head, grids, indices, arm, chunk):
    h, w = grids.shape[1:3]; radius = head.depth if isinstance(head, SpatialAdapter) else 0
    result = 0.
    for part in indices.split(chunk):
        tuples = part//(h*w); clean_view = tuples//8*5; shift_view = clean_view+1+tuples%4
        rows, cols = part%(h*w)//w, part%w
        z, valid = gather_neighborhoods(grids, clean_view, rows, cols, radius)
        zt, valid_t = gather_neighborhoods(grids, shift_view, rows, cols, radius)
        clean, shifted = center_features(head, z, valid), center_features(head, zt, valid_t)
        teacher = z[:, :, radius, radius].detach()
        align = shifted-teacher if arm == 'L2' else shifted-clean
        anchor_weight = .2 if arm == 'L1' else 0. if arm == 'A1-no-anchor' else .1
        illumination = align.square().sum(-1).sum()/len(indices)
        anchor = (clean-teacher).square().sum(-1).sum()/len(indices)
        loss = illumination + anchor_weight*anchor
        assert torch.isfinite(loss)
        loss.backward(); result += float(loss.detach())
    return result


def synthetic_cache(model, sub, data, cfg):
    """Only the reported cable A2/A3 pilot uses synthetic-defect views."""
    from PIL import Image
    from torchvision.transforms import functional as TF, InterpolationMode
    from src.defect_adapter import quadruplets
    from src.features import MEAN, STD
    views = [[], [], [], []]
    with torch.inference_mode():
        for name in sub['fit_paths']:
            x = TF.to_tensor(TF.resize(Image.fromarray(rgb(data/name)), 448, InterpolationMode.BICUBIC, antialias=True))
            for quadruplet, _, _ in quadruplets(x, name, sub['seed'], cfg['training_conditions']):
                for dest, image in zip(views, quadruplet):
                    z = model.get_intermediate_layers(TF.normalize(image, MEAN, STD)[None].cuda())[0][0].contiguous()
                    dest.append(normalize(z))
    return [torch.cat(v).cuda() for v in views]


def diagnostics(head, grids):
    illumination, anchor = [], []
    with torch.inference_mode():
        for start in range(0, len(grids), 5):
            z = grids[start]; clean = transform_grid(head, z)
            anchor.append((clean-z).square().sum(-1).double().mean().item())
            for index in range(1, 5):
                shifted = transform_grid(head, grids[start+index])
                illumination.append((shifted-clean).square().sum(-1).double().mean().item())
    return {'illumination_squared_l2': float(np.mean(illumination)),
            'anchor_squared_l2': float(np.mean(anchor)), 'diagnostic_images': len(grids)//5}


def train(model, sub, arm, data, cfg, out, chunk):
    from src.defect_adapter import loss_components, objective
    name = f"{sub['category']}_s{sub['seed']}_{arm}"
    head = head_for(arm, sub['seed'])
    grids = normal_grids(model, sub['fit_paths'], data, cfg, out)
    indices, hashes = schedules(sub, cfg, grids.shape[1:3], full=arm in ('A2', 'A3'))
    ids = torch.from_numpy(indices).cuda()
    extra = synthetic_cache(model, sub, data, cfg) if arm in ('A2', 'A3') else None
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001, foreach=False)
    torch.cuda.reset_peak_memory_stats(); started = sync(); losses = []
    for step, indices in enumerate(ids, 1):
        optimizer.zero_grad(set_to_none=True)
        if extra is not None:
            parts = loss_components(head, [v[indices] for v in extra], 512)
            loss = objective(parts, arm)
            assert torch.isfinite(loss); loss.backward(); value = float(loss.detach())
        else:
            value = backward_centers(head, grids, indices, arm, chunk)
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head.parameters())
        optimizer.step()
        if step in (1, 100, 250, 500, 750, 1000):
            losses.append({'step': step, 'loss': value})
    seconds = sync()-started
    cp = out/'checkpoints'/f'{name}.pt'; cp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), cp)
    record = {'arm': arm, 'category': sub['category'], 'seed': sub['seed'], 'shots': sub['shots'],
              'parameters': sum(p.numel() for p in head.parameters()), 'steps': 1000, 'fit_s': seconds,
              'checkpoint': str(cp.relative_to(out)), 'checkpoint_sha256': sha256(cp),
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'loss_log': losses, **hashes}
    head.eval().requires_grad_(False)
    heldout = normal_grids(model, sub['diagnostic_paths'], data, cfg, out)
    record['diagnostic'] = diagnostics(head, heldout)
    record['A0_diagnostic'] = diagnostics(None, heldout)
    del heldout
    save(out/'training'/f'{name}.json', record)
    print('Trained', name, flush=True)
    del head, optimizer, grids, ids, extra; torch.cuda.empty_cache()


def evaluate(model, sub, arm, data, cfg, out):
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from sklearn.metrics import roc_auc_score
    from src.evaluation import pixel_average_precision
    name = f"{sub['category']}_s{sub['seed']}_{arm}"
    head = head_for(arm, sub['seed'])
    if head is not None:
        record = read(out/'training'/f'{name}.json'); cp = out/record['checkpoint']
        assert record['steps'] == 1000 and sha256(cp) == record['checkpoint_sha256']
        head.load_state_dict(torch.load(cp, weights_only=True)); head.eval().requires_grad_(False)
    with torch.inference_mode():
        vectors = []
        for path in sub['reference_paths']:
            for image in ad.rotations(rgb(data/path)):
                vectors.append(transform_grid(head, encode(model, image, cfg['conditions'][0]).cuda()).reshape(-1,384))
        bank = torch.cat(vectors); del vectors
        assert bank.shape == (sub['shots']*8192, 384)
        paths = sorted((data/sub['category']/'test').glob('*/*.png'))
        masks = np.zeros((len(paths),224,224), dtype=bool); labels = []
        for i, p in enumerate(paths):
            labels.append(p.parent.name != 'good')
            if labels[-1]:
                mp = data/sub['category']/'ground_truth'/p.parent.name/(p.stem+'_mask.png')
                with Image.open(mp) as image:
                    masks[i] = np.asarray(image.convert('L').resize((224,224), Image.Resampling.NEAREST))>0
        rows = []
        for condition in cfg['conditions']:
            maps, scores, patches = [], [], []; scoring_s = 0.
            torch.cuda.reset_peak_memory_stats(); total_started = sync()
            for path in paths:
                image = rgb(path); grid = encode(model, image, condition).cuda()
                started = sync()
                q = transform_grid(head, grid).reshape(-1,384)
                d = ad.distances(q, bank, cfg['search_chunk']).cpu().numpy()
                scoring_s += sync()-started
                patches.append(d); scores.append(ad.image_score(d))
                native = cv2.resize(d.astype(float).reshape(grid.shape[:2]), image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
                maps.append(cv2.resize(gaussian_filter(native, sigma=4), (224,224), interpolation=cv2.INTER_LINEAR))
            elapsed = sync()-total_started
            ap = pixel_average_precision(torch.from_numpy(masks).cuda(), torch.from_numpy(np.stack(maps)).cuda())
            dest = out/'patches'/name/(condition['name']+'.npy'); dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, np.stack(patches))
            rows.append({'method': arm, 'arm': arm, 'category': sub['category'], 'seed': sub['seed'],
                         'shots': sub['shots'], 'condition': condition['name'], 'pixel_ap': ap,
                         'image_auroc': roc_auc_score(labels,scores), 'n_test': len(paths),
                         'cached_feature_scoring_s': scoring_s, 'evaluation_s_including_encoder_and_mapping': elapsed,
                         'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                         'bank_vectors': len(bank), 'patch_sha256': sha256(dest)})
            scores_file = dest.with_suffix('.csv')
            write_csv(scores_file,[{'path':str(p.relative_to(data)), 'score':s} for p,s in zip(paths,scores)])
        write_csv(out/'units'/f'{name}.csv', rows)
    print('Evaluated', name, flush=True)
    del bank, head; torch.cuda.empty_cache()


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-schedules', action='store_true')
    p.add_argument('--data', type=Path, default=Path('data/mvtec_ad'))
    p.add_argument('--out', type=Path)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=['A0-16','MLP32-16'])
    p.add_argument('--categories', nargs='+', choices=protocol()['categories'], default=protocol()['categories'])
    p.add_argument('--seeds', nargs='+', type=int, choices=[0,1,2], default=[0,1,2])
    p.add_argument('--chunk', type=int, default=512, help='Gradient accumulation chunk; denominator remains 512.')
    args = p.parse_args(); os.chdir(ROOT)
    if args.check_schedules:
        verify_schedules(); return
    assert args.out is not None and not args.out.exists(), 'Supply a new --out directory; failed runs are preserved.'
    assert 1 <= args.chunk <= 512
    assert len(args.arms)==len(set(args.arms)) and len(args.categories)==len(set(args.categories)) and len(args.seeds)==len(set(args.seeds))
    if set(args.arms)&{'A2','A3'}:
        assert args.categories==['cable'], 'A2/A3 are restricted to the reported cable pilot.'
    if 'A1-no-anchor' in args.arms:
        assert set(args.categories) <= {'bottle','cable','carpet'}, 'No-anchor scope is the reported three-category study.'
    cfg = protocol()
    if set(args.arms)&{'A2','A3'}:
        cfg['conditions'] = read(ROOT/'configs/defect_adapter_pilot.json')['conditions']
    args.out.mkdir(parents=True)
    run = {'status':'checking', 'release_orchestration':True, 'arms':args.arms, 'categories':args.categories,
           'seeds':args.seeds, 'chunk':args.chunk, 'source_sha256':sha256(Path(__file__)),
           'protocol_sha256':sha256(ROOT/'configs/a1_capacity_data_cnn.json')}
    save(args.out/'run.json',run)
    try:
        dependencies(); verify_data(args.data,args.categories)
        encoder_receipt = read(ROOT/'cache/encoder_source.json')
        assert encoder_receipt['revision'] == ad.DINO_SHA
        for name, digest in encoder_receipt['files'].items():
            assert sha256(ROOT/'cache/upstream/dinov2'/name) == digest, name
        run['encoder_source_receipt_sha256'] = sha256(ROOT/'cache/encoder_source.json')
        weights = ROOT/'cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
        assert sha256(weights)==cfg['weights_sha256']
        run['python']=sys.version; run['torch']=torch.__version__; run['gpu']=torch.cuda.get_device_name()
        manifest=read(ROOT/'results/metrics/a1_capacity_data_cnn_20261111/splits.json')
        units=[]
        for cat in args.categories:
            for seed in args.seeds:
                for arm in args.arms:
                    shots=32 if arm.endswith('-32') else 16
                    sub=next(s for s in manifest['subsets'] if (s['category'],s['seed'],s['shots'])==(cat,seed,shots))
                    schedules(sub,cfg); units.append((sub,arm))
        save(args.out/'config.json',cfg); save(args.out/'subsets.json',[s for s,a in units])
        model=ad.load(); run['status']='training'; save(args.out/'run.json',run)
        # Finish every fixed final-step checkpoint before accessing test scores.
        for sub,arm in units:
            if not arm.startswith('A0'):
                train(model,sub,arm,args.data,cfg,args.out,args.chunk)
        run['status']='evaluating'; save(args.out/'run.json',run)
        for sub,arm in units:
            evaluate(model,sub,arm,args.data,cfg,args.out)
        rows=[]
        for file in sorted((args.out/'units').glob('*.csv')):
            with file.open(newline='') as stream:
                rows.extend(csv.DictReader(stream))
        assert len(rows)==len(units)*len(cfg['conditions'])
        write_csv(args.out/'metrics.csv',rows); run['status']='completed'; run['evaluations']=len(rows)
    except BaseException:
        run['status']='failed'; run['error']=traceback.format_exc(); raise
    finally:
        save(args.out/'run.json',run)


if __name__=='__main__':
    main()
