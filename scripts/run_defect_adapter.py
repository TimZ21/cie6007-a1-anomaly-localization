"""Fixed normal-only fitting, then separate A0-A3 pilot evaluation.

No test image is opened in fit mode. Final-step heads are frozen before the
evaluate process starts. Existing project code, manifests and runs are read-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import cv2
import faiss
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torchvision.transforms import functional as TF, InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_assets import sha256
from run_e1 import save, utc
from run_e2 import normalized, verify_upstream
from run_comparison import sync_time
from run_backbone_comparison import setup, rgb, references, test_data, resources, gpu_snapshot
from src import anomalydino as ad
from src.defect_adapter import DefectAdapter, quadruplets, loss_components, objective, batch_indices
from src.evaluation import pixel_average_precision

CONFIG = Path('configs/defect_adapter_pilot.json')
SOURCES = ['src/defect_adapter.py', 'tests/test_defect_adapter.py', 'scripts/run_defect_adapter.py',
           'scripts/analyze_defect_adapter.py', 'scripts/run_backbone_comparison.py', 'scripts/run_e2.py',
           'scripts/run_e1.py', 'scripts/run_comparison.py', 'scripts/prepare_assets.py',
           'src/anomalydino.py', 'src/features.py', 'src/banks.py', 'src/evaluation.py']
VIEWS = ['normal', 'shift', 'defect', 'shift_defect']


def identity(cfg):
    checkpoint = Path('cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')
    assert sha256(checkpoint) == cfg['weights_sha256']
    archives = {}
    for folder, filename in [('dinov2', 'dinov2-pinned.zip'), ('AnomalyDINO', 'anomalydino-pinned.zip')]:
        archive = Path('cache') / filename; archives[filename] = sha256(archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if not name.endswith('/'):
                    assert (Path('cache/upstream') / folder / name).read_bytes() == z.read(name)
    return {'source_sha256': {f: sha256(Path(f)) for f in SOURCES}, 'config_sha256': sha256(CONFIG),
            'split_sha256': sha256(Path('splits/mvtec_ad_5seed.json')),
            'inventory_sha256': sha256(Path('assets/data_files.json')),
            'weights_sha256': sha256(checkpoint), 'upstream_archives_sha256': archives}


def selected(cfg, ident):
    split = json.loads(Path('splits/mvtec_ad_5seed.json').read_text())
    assert split['inventory_sha256'] == ident['inventory_sha256']
    subsets = [s for s in split['subsets'] if s['category'] in cfg['categories']
               and s['shots'] in cfg['shots'] and s['seed'] in cfg['seeds']]
    assert len(subsets) == 3 and cfg['categories'] == ['cable'] and cfg['shots'] == [16]
    return sorted(subsets, key=lambda x: x['seed']), {
        r['path']: r for r in json.loads(Path('assets/data_files.json').read_text())}


def state_hash(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def partition(sub, cfg):
    names = [r['path'] for r in sub['train']]
    indices = np.random.default_rng(cfg['training']['split_seed_offset'] + sub['seed']).permutation(16)
    ordered = [names[i] for i in indices]
    return ordered[:12], ordered[12:]


@torch.inference_mode()
def encode_rgb(model, x):
    tokens = model.get_intermediate_layers(TF.normalize(x, ad.MEAN, ad.STD)[None].cuda())[0][0]
    return normalized(tokens).cpu().numpy().copy()


def build_cache(model, paths, seed, cfg, inventory, dest):
    count = len(paths) * 2 * len(cfg['training_conditions'])
    arrays = {v: np.empty((count, 1024, 384), dtype=np.float32) for v in VIEWS}
    masks = np.empty((count, 1024), dtype=bool); records = []; index = 0
    for name in paths:
        path = Path('data/mvtec_ad') / name
        assert '/train/good/' in name and sha256(path) == inventory[name]['sha256']
        im = TF.to_tensor(TF.resize(Image.fromarray(rgb(path)), 448, InterpolationMode.BICUBIC, antialias=True))
        normal = encode_rgb(model, im); shifts = {}
        for views, mask, meta in quadruplets(im, name, seed, cfg['training_conditions']):
            cname = meta['condition']['name']
            if cname not in shifts:
                shifts[cname] = encode_rgb(model, views[1])
            features = [normal, shifts[cname], encode_rgb(model, views[2]), encode_rgb(model, views[3])]
            for v, feature in zip(VIEWS, features):
                arrays[v][index] = feature
            masks[index] = mask.numpy(); records.append(meta); index += 1
    np.savez(dest, **arrays, interior=masks)
    return arrays, masks, records


def view_tensors(arrays, start, end):
    return [torch.from_numpy(arrays[v][start:end].reshape(-1, 384)) for v in VIEWS]


@torch.inference_mode()
def diagnostics(adapter, arrays, mask):
    values = {'illumination_squared_l2': [], 'anchor_squared_l2': [],
              'clean_defect_squared_error': [], 'cross_defect_squared_error': [],
              'teacher_defect_norm': [], 'cross_defect_norm': [], 'cross_defect_cosine': []}
    flat = [torch.from_numpy(arrays[v].reshape(-1, 384)) for v in VIEWS]
    valid = torch.from_numpy(mask.flatten())
    for start in range(0, len(flat[0]), 2048):
        z, zt, za, zta = [x[start:start + 2048].cuda() for x in flat]
        h, ht, ha, hta = [adapter(x) if adapter is not None else x for x in [z, zt, za, zta]]
        values['illumination_squared_l2'].append((ht-h).square().sum(-1).cpu())
        values['anchor_squared_l2'].append((h-z).square().sum(-1).cpu())
        sel = valid[start:start + 2048].cuda()
        if sel.any():
            target, clean, cross = (za-z)[sel], (ha-h)[sel], (hta-ht)[sel]
            tn, cn = target.norm(dim=-1), cross.norm(dim=-1)
            values['clean_defect_squared_error'].append((clean-target).square().sum(-1).cpu())
            values['cross_defect_squared_error'].append((cross-target).square().sum(-1).cpu())
            values['teacher_defect_norm'].append(tn.cpu()); values['cross_defect_norm'].append(cn.cpu())
            useful = tn > 1e-4
            if useful.any():
                values['cross_defect_cosine'].append(torch.nn.functional.cosine_similarity(cross[useful], target[useful], dim=-1).cpu())
    result = {k: float(torch.cat(v).double().mean()) for k, v in values.items()}
    result['cross_to_teacher_mean_norm_ratio'] = result['cross_defect_norm'] / max(result['teacher_defect_norm'], 1e-12)
    result['interior_patches'] = int(valid.sum())
    return result


def fit_one(method, seed, cache, masks, cfg, local):
    c = cfg['training']; init_seed = c['initialization_seed_offset'] + seed
    torch.manual_seed(init_seed)
    head = DefectAdapter(bottleneck=c['bottleneck']).cuda()
    initial_sha = state_hash(head)
    data = view_tensors(cache, 0, 12 * 8)
    interior = np.flatnonzero(masks[:12 * 8].flatten())
    rng = np.random.default_rng(c['batch_seed_offset'] + seed); schedule = hashlib.sha256()
    opt = torch.optim.AdamW(head.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'], foreach=False)
    record = {'method': method, 'seed': seed, 'initial_state_sha256': initial_sha,
              'parameters': sum(p.numel() for p in head.parameters()), 'loss_log': []}
    start = sync_time()
    for step in range(1, c['steps'] + 1):
        ids = batch_indices(rng, len(data[0]), interior, c['batch_size'])
        schedule.update(ids.astype('<i8').tobytes())
        batch = [x[ids].cuda() for x in data]
        opt.zero_grad(set_to_none=True)
        parts = loss_components(head, batch, c['batch_size'] // 2)
        loss = objective(parts, method, c['anchor_weight'], c['defect_weight'])
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
        if step in (1, 100, 250, 500, 750, c['steps']):
            record['loss_log'].append({'step': step, 'total': float(loss.detach()),
                                      **{k: float(v.detach()) for k, v in parts.items()}})
    record['fit_s'] = sync_time() - start; record['batch_schedule_sha256'] = schedule.hexdigest()
    record['final_state_sha256'] = state_hash(head); head.eval().requires_grad_(False)
    checkpoint = local / f'cable_s{seed}_{method}.pt'
    torch.save(head.state_dict(), checkpoint)
    record.update(checkpoint=checkpoint.as_posix(), checkpoint_sha256=sha256(checkpoint),
                  checkpoint_step=c['steps'], initializer_seed=init_seed,
                  diagnostic=diagnostics(head, {k: v[96:] for k, v in cache.items()}, masks[96:]))
    return record


def environment():
    return {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'numpy': np.__version__,
            'faiss': faiss.__version__, 'opencv': cv2.__version__, 'tf32': False, 'autocast': False,
            'precision': 'float32 frozen features/adapter/search; float64 maps/AP reduction'}


def fit(cfg, out, local, ident):
    assert not out.exists() and not local.exists(), 'Preserve all previous attempts/results'
    out.mkdir(parents=True); local.mkdir(parents=True); (out / 'training').mkdir(); (out / 'units').mkdir()
    (out / 'config.json').write_bytes(CONFIG.read_bytes())
    rec = {**ident, 'run_id': out.name, 'status': 'fitting', 'fit_started_utc': utc(),
           'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
           'base_commit': cfg['base_commit'], 'parent_commit': cfg['parent_commit'],
           'environment': environment(), 'scope': cfg['scope'], 'completed_training_sets': 0,
           'completed_evaluations': 0, 'expected_evaluations': 36, 'pid': os.getpid(),
           'nvidia_before_fit': gpu_snapshot(), 'training_units': []}
    save(out / 'run.json', rec); start = time.perf_counter()
    try:
        subsets, inv = selected(cfg, ident); model = ad.load()
        assert all(not p.requires_grad for p in model.parameters())
        torch.cuda.reset_peak_memory_stats()
        # Real normal-only numerical and full-bank memory preflight.
        names = [r['path'] for r in subsets[0]['train']]
        refs = references(model, names, 384, cfg['conditions'][0], inv)
        bank = torch.cat([refs[n] for n in names]).cuda()
        normal = rgb(Path('data/mvtec_ad') / names[0])
        query = normalized(ad.features(model, normal, cfg['conditions'][1]))
        check = verify_upstream(model, normal, bank, query)
        torch.manual_seed(5000); initial = DefectAdapter().cuda().eval()
        with torch.inference_mode():
            init_error = float((ad.distances(initial(query), initial(bank)) - ad.distances(query, bank)).abs().max())
        assert init_error < 2e-6
        rec['normal_preflight'] = {**check, 'zero_adapter_distance_max_error': init_error, **resources()}
        save(out / 'run.json', rec)
        del refs, bank, query, initial
        for sub in subsets:
            seed = sub['seed']; fit_names, diagnostic_names = partition(sub, cfg)
            assert len(set(fit_names) & set(diagnostic_names)) == 0
            dest = local / f'cable_s{seed}_quadruplets.npz'; t = sync_time()
            cache, masks, quadruplet_records = build_cache(model, fit_names + diagnostic_names, seed, cfg, inv, dest)
            unit = {'category': 'cable', 'seed': seed, 'shots': 16,
                    'reference_paths': [r['path'] for r in sub['train']], 'fit_paths': fit_names,
                    'diagnostic_paths': diagnostic_names, 'cache': dest.as_posix(), 'cache_sha256': sha256(dest),
                    'cache_shape_per_view': list(cache['normal'].shape), 'cache_extraction_s': sync_time() - t,
                    'quadruplets': quadruplet_records, 'status': 'fitting', 'models': [],
                    'A0_diagnostic': diagnostics(None, {k: v[96:] for k, v in cache.items()}, masks[96:])}
            unit_path = out / 'training' / f'cable_s{seed}.json'; save(unit_path, unit)
            # Rotate order to reduce systematic runtime order effects; paired seeds stay identical.
            order = ['A1', 'A2', 'A3']; order = order[seed:] + order[:seed]
            unit['method_order'] = order
            for method in order:
                model_record = fit_one(method, seed, cache, masks, cfg, local)
                unit['models'].append(model_record); save(unit_path, unit)
                print(f'cable seed={seed} {method}: 1000 fixed steps, {model_record["fit_s"]:.1f}s; no test data opened', flush=True)
            assert len({m['initial_state_sha256'] for m in unit['models']}) == 1
            assert len({m['batch_schedule_sha256'] for m in unit['models']}) == 1
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            unit.update(status='fitted', fit_finished_utc=utc()); save(unit_path, unit)
            rec['training_units'].append({'path': unit_path.as_posix(), 'sha256': sha256(unit_path)})
            rec['completed_training_sets'] += 1; save(out / 'run.json', rec)
            del cache, masks; gc.collect(); torch.cuda.empty_cache()
        rec.update(status='fitted', fit_resources=resources(), nvidia_after_fit=gpu_snapshot())
        assert rec['fit_resources']['peak_cuda_reserved_mib'] < 6144
    except BaseException as e:
        rec.update(status='fit_failed', error=repr(e)); raise
    finally:
        rec.update(fit_finished_utc=utc(), fit_elapsed_s=time.perf_counter() - start); save(out / 'run.json', rec)


@torch.inference_mode()
def transform(head, values):
    if head is None:
        return values
    return torch.cat([head(x) for x in values.split(8192)])


def previous_check(key, rows, metrics):
    folder = Path('results/metrics/e2_anomalydino_20261023/units')
    with (folder / (key + '_scores.csv')).open(newline='') as f:
        old = {(r['condition'], r['path']): float(r['score']) for r in csv.DictReader(f)}
    score_error = max(abs(r['score'] - old[(r['condition'], r['path'])]) for r in rows if r['method'] == 'A0')
    oldmetrics = json.loads((folder / (key + '.json')).read_text())['metrics']
    errors = {'image_score_max_error': score_error, 'pixel_ap_max_error': 0., 'image_auroc_max_error': 0.}
    for m in metrics:
        if m['method'] == 'A0':
            oldm = next(r for r in oldmetrics if r['condition'] == m['condition'])
            for field in ['pixel_ap', 'image_auroc']:
                errors[field + '_max_error'] = max(errors[field + '_max_error'], abs(m[field] - oldm[field]))
    assert errors['image_score_max_error'] < 2e-5 and errors['pixel_ap_max_error'] < 2e-6 and errors['image_auroc_max_error'] < 1e-10
    return errors


@torch.inference_mode()
def evaluate(cfg, out, local, ident):
    rec = json.loads((out / 'run.json').read_text())
    assert rec['status'] == 'fitted' and rec['completed_training_sets'] == 3
    assert all(rec[k] == v for k, v in ident.items())
    assert rec['source_commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    training = {}
    for r in rec['training_units']:
        assert sha256(Path(r['path'])) == r['sha256']
        u = json.loads(Path(r['path']).read_text()); training[u['seed']] = u
        for m in u['models']:
            assert sha256(Path(m['checkpoint'])) == m['checkpoint_sha256']
    rec.update(status='evaluating', evaluation_started_utc=utc(), nvidia_before_eval=gpu_snapshot(), pid=os.getpid())
    save(out / 'run.json', rec); start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    try:
        subsets, inv = selected(cfg, ident); model = ad.load()
        names = sorted({r['path'] for s in subsets for r in s['train']})
        t = sync_time(); refs = references(model, names, 384, cfg['conditions'][0], inv)
        rec['reference_extraction_s'] = sync_time() - t
        # This is the first access to test images/labels in the entire run.
        paths, test_names, labels, masks, shapes = test_data('cable', inv)
        tests = {}; t = sync_time()
        for condition in cfg['conditions']:
            tests[condition['name']] = torch.stack([normalized(ad.features(model, rgb(p), condition)).cpu() for p in paths])
            print('Extracted test features: ' + condition['name'], flush=True)
        rec['test_extraction_s'] = sync_time() - t; gpu_masks = torch.from_numpy(masks).cuda()
        with ThreadPoolExecutor(max_workers=4) as workers:
            for sub in subsets:
                seed = sub['seed']; key = f'cable_s{seed}_k16'; tunit = sync_time()
                raw_bank = torch.cat([refs[r['path']] for r in sub['train']]).cuda()
                assert raw_bank.shape == (131072, 384)
                unit = {'category': 'cable', 'seed': seed, 'shots': 16, 'key': key,
                        'reference_paths': [r['path'] for r in sub['train']], 'bank_vectors': len(raw_bank),
                        'bank_bytes': raw_bank.numel() * 4, 'raw_bank_sha256': hashlib.sha256(raw_bank.cpu().numpy().tobytes()).hexdigest(),
                        'source_commit': rec['source_commit'], 'config_sha256': rec['config_sha256'], 'metrics': [], 'banks': {}}
                rows = []; order = cfg['methods'][seed:] + cfg['methods'][:seed]; unit['method_order'] = order
                for method in order:
                    head = None
                    if method != 'A0':
                        info = next(m for m in training[seed]['models'] if m['method'] == method)
                        head = DefectAdapter(bottleneck=cfg['training']['bottleneck']).cuda()
                        head.load_state_dict(torch.load(info['checkpoint'], map_location='cuda', weights_only=True))
                        head.eval().requires_grad_(False); assert state_hash(head) == info['final_state_sha256']
                    t = sync_time(); bank = transform(head, raw_bank); bank_s = sync_time() - t
                    bank_file = local / f'{key}_{method}_bank.npy'; np.save(bank_file, bank.cpu().numpy())
                    unit['banks'][method] = {'file': bank_file.as_posix(), 'sha256': sha256(bank_file), 'transform_s': bank_s}
                    ad.distances(transform(head, tests['clean'][0].cuda()), bank); sync_time()
                    for condition in cfg['conditions']:
                        cname = condition['name']; patches = []; scoring_s = 0.; adapter_s = 0.
                        for i, name in enumerate(test_names):
                            q = tests[cname][i].cuda(); t = sync_time()
                            q = transform(head, q); middle = sync_time()
                            d = ad.distances(q, bank, cfg['search_chunk']); end = sync_time()
                            adapter_s += middle - t; scoring_s += end - t
                            patch = d.cpu().numpy(); assert np.isfinite(patch).all(); patches.append(patch)
                            rows.append({'condition': cname, 'method': method, 'path': name, 'label': int(labels[i]), 'score': ad.image_score(patch)})
                        patch = np.stack(patches); pp = local / f'{key}_{method}_{cname}_patches.npy'; np.save(pp, patch)
                        t = time.perf_counter(); maps = np.stack(list(workers.map(ad.map224, patch, shapes))); mapping_s = time.perf_counter() - t
                        scores = [r['score'] for r in rows if r['condition'] == cname and r['method'] == method]
                        unit['metrics'].append({'category': 'cable', 'seed': seed, 'shots': 16, 'method': method, 'condition': cname,
                            'image_auroc': float(roc_auc_score(labels, scores)),
                            'pixel_ap': pixel_average_precision(gpu_masks, torch.from_numpy(maps).cuda()),
                            'n_test': len(paths), 'n_anomaly': int(labels.sum()), 'scoring_s': scoring_s,
                            'adapter_s': adapter_s, 'mapping_s': mapping_s, 'patch_file': pp.as_posix(), 'patch_sha256': sha256(pp)})
                        del maps
                    print(f'{key} {method}: all three conditions complete', flush=True)
                    del bank, head
                sp = out / 'units' / (key + '_scores.csv')
                with sp.open('w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=['condition', 'method', 'path', 'label', 'score'], lineterminator='\n'); w.writeheader(); w.writerows(rows)
                unit.update(status='completed', elapsed_s=sync_time() - tunit, image_scores_file=sp.name,
                            image_scores_sha256=sha256(sp), previous_e2_reproduction=previous_check(key, rows, unit['metrics']))
                save(out / 'units' / (key + '.json'), unit)
                rec['completed_evaluations'] += len(unit['metrics']); save(out / 'run.json', rec)
                del raw_bank
        assert rec['completed_evaluations'] == 36
        rec.update(status='completed', eval_resources=resources(), nvidia_after_eval=gpu_snapshot())
    except BaseException as e:
        rec.update(status='evaluation_failed', error=repr(e)); raise
    finally:
        rec.update(evaluation_finished_utc=utc(), evaluation_elapsed_s=time.perf_counter() - start)
        rec['total_fit_and_evaluation_s'] = rec['fit_elapsed_s'] + rec['evaluation_elapsed_s']
        save(out / 'run.json', rec)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['fit', 'evaluate'], required=True)
    p.add_argument('--run-id', default='defect_adapter_20261102'); args = p.parse_args()
    setup(); torch.use_deterministic_algorithms(True)
    cfg = json.loads(CONFIG.read_text()); ident = identity(cfg)
    assert subprocess.check_output(['git', 'rev-parse', 'main'], text=True).strip() == cfg['base_commit']
    assert subprocess.check_output(['git', 'rev-parse', 'codex/cross-reference-calibration'], text=True).strip() == cfg['parent_commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    out, local = Path('results/metrics') / args.run_id, Path('runs') / args.run_id
    (fit if args.mode == 'fit' else evaluate)(cfg, out, local, ident)


if __name__ == '__main__':
    main()
