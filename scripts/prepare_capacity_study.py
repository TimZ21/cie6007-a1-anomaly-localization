"""Fixed normal-only splits and center schedules for the A1 capacity study.

The old 16-shot identities are read, never regenerated or overwritten. The
legacy inactive draws are reproduced solely to preserve the existing A1 center
schedule; no synthetic image or defect loss is used by the new experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('configs/a1_capacity_data_cnn.json')
DEVELOPMENT = ('bottle', 'cable', 'carpet')
ARMS = ('A0-16', 'A0-32', 'MLP32-16', 'MLP64-16', 'MLP32-32',
        'MLP64-32', 'CNN1-32', 'CNN3-32')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit LF keeps the identity equal to the committed Git blob on Windows.
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def origin(category, cfg):
    return Path(cfg['reference_runs']['development' if category in DEVELOPMENT else 'extension'])


def validate_config(cfg, root=ROOT):
    old = json.loads((root / cfg['reference_runs']['development'] / 'config.json').read_text())
    for key in ('backbone', 'input_size', 'rotations', 'search_chunk', 'weights_sha256',
                'conditions', 'training_conditions', 'seeds'):
        assert cfg[key] == old[key], ('Changed frozen setting', key)
    for key in ('steps', 'learning_rate', 'weight_decay', 'anchor_weight',
                'initialization_seed_offset', 'batch_seed_offset', 'split_seed_offset'):
        assert cfg['training'][key] == old['training'][key], ('Changed training setting', key)
    assert cfg['training']['effective_center_patches'] == old['training']['batch_size'] // 2 == 512
    assert cfg['training']['legacy_batch_size'] == old['training']['batch_size'] == 1024
    assert cfg['arms'] == list(ARMS)
    assert cfg['shots'] == [16, 32] and cfg['seeds'] == [0, 1, 2]
    assert len(cfg['categories']) == len(set(cfg['categories'])) == 15
    assert set(DEVELOPMENT).issubset(cfg['categories'])
    assert len(cfg['conditions']) == 7 and cfg['conditions'][0]['name'] == 'clean'
    assert cfg['expected_evaluations'] == 8 * 15 * 3 * 7 == 2520
    assert cfg['expected_reused_evaluations'] == 2 * 15 * 3 * 7 == 630
    assert cfg['expected_new_evaluations'] == 6 * 15 * 3 * 7 == 1890
    assert cfg['expected_new_heads'] == 5 * 15 * 3 == 225


def additional_paths(names, existing, seed, count=16):
    """PCG64(seed + 3000) permutation of lexicographically sorted unused normals."""
    remaining = sorted(set(names).difference(existing))
    assert len(remaining) >= count and len(set(existing)) == len(existing)
    order = np.random.Generator(np.random.PCG64(3000 + seed)).permutation(len(remaining))[:count]
    return [remaining[int(i)] for i in order]


def prepare(cfg, root=ROOT):
    root = Path(root)
    validate_config(cfg, root)
    inventory_path = Path(cfg['inventory'])
    inventory = json.loads((root / inventory_path).read_text())
    lookup = {r['path']: r for r in inventory}
    assert len(lookup) == len(inventory)
    validation = json.loads((root / cfg['data_validation']).read_text())
    assert validation['status'] == 'passed'
    assert validation['inventory_sha256'] == sha256(root / inventory_path)
    assert set(validation['categories']) == set(cfg['categories'])
    subsets, origins = [], {}
    for cat in cfg['categories']:
        normal_names = sorted(p for p in lookup if p.startswith(cat + '/train/good/'))
        assert len(normal_names) == validation['categories'][cat]['train_good']
        for seed in cfg['seeds']:
            base = origin(cat, cfg)
            training_file = base / 'training' / f'{cat}_s{seed}.json'
            unit_file = base / 'units' / f'{cat}_s{seed}_k16.json'
            training = json.loads((root / training_file).read_text())
            unit = json.loads((root / unit_file).read_text())
            assert training['status'] == 'fitted' and unit['status'] == 'completed'
            refs = training['reference_paths']
            fit, diag = training['fit_paths'], training['diagnostic_paths']
            assert len(refs) == len(set(refs)) == 16
            assert len(fit) == 12 and len(diag) == 4
            assert not set(fit) & set(diag) and set(fit + diag) == set(refs)
            assert unit['reference_paths'] == refs
            assert set(refs).issubset(normal_names)
            extra = additional_paths(normal_names, refs, seed)
            source = {'original_training_file': training_file.as_posix(),
                      'original_training_sha256': sha256(root / training_file),
                      'original_unit_file': unit_file.as_posix(),
                      'original_unit_sha256': sha256(root / unit_file)}
            origins[f'{cat}_s{seed}'] = source
            for shots in cfg['shots']:
                paths = refs if shots == 16 else refs + extra
                fitting = fit if shots == 16 else fit + extra[:12]
                diagnostics = diag if shots == 16 else diag + extra[12:]
                assert len(paths) == len(set(paths)) == shots
                assert len(fitting) == shots * 3 // 4 and len(diagnostics) == shots // 4
                assert set(fitting).isdisjoint(diagnostics)
                assert set(fitting + diagnostics) == set(paths)
                subsets.append({'category': cat, 'seed': seed, 'shots': shots,
                    'reference_paths': paths, 'fit_paths': fitting, 'diagnostic_paths': diagnostics,
                    'train': [{'path': p, 'sha256': lookup[p]['sha256']} for p in paths],
                    'additional_paths': [] if shots == 16 else extra, **source})
    assert len(subsets) == 90
    return {'schema_version': 1, 'purpose': 'A1 capacity/data/CNN stages 1 and 2 only',
        'generator': 'Preserve recorded 16; PCG64(seed+3000) permutation of sorted unused official normal training paths; first12 extra fit, last4 extra diagnostic.',
        'inventory': inventory_path.as_posix(), 'inventory_sha256': sha256(root / inventory_path),
        'data_validation_sha256': sha256(root / cfg['data_validation']),
        'numpy_version': np.__version__, 'paired_arms': cfg['arms'], 'seeds': cfg['seeds'],
        'nested_shots': cfg['shots'], 'training_origins': origins, 'subsets': subsets}


def rectangle_metadata(path, seed, conditions, image_hw=(448, 448)):
    """Reproduce only the legacy random metadata, without creating synthetic RGB."""
    height_px, width_px = image_hw
    image_seed = int.from_bytes(hashlib.sha256(f'{seed}:{path}'.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(image_seed)
    result = []
    for kind in ('cutpaste', 'local_color'):
        for condition in conditions:
            height, width = (int(x) for x in rng.integers(84, 141, size=2))
            top = int(rng.integers(14, height_px - height - 14 + 1))
            left = int(rng.integers(14, width_px - width - 14 + 1))
            row = {'path': path, 'kind': kind, 'condition': condition,
                   'rectangle_tlhw': [top, left, height, width]}
            if kind == 'cutpaste':
                sy = int(rng.integers(0, height_px - height + 1))
                sx = int(rng.integers(0, width_px - width + 1))
                row['donor_top_left'] = [sy, sx]
            else:
                row['rgb_offset'] = (np.asarray(rng.choice([-1., 1.], size=3),
                    dtype=np.float32) * np.float32(.18)).tolist()
            result.append(row)
    return result


def interior_indices(records, grid_hw):
    """Exactly match avg_pool2d(binary interior,14,14) >= .8 via integer areas."""
    height, width = (int(x) for x in grid_hw)
    yy, xx = np.arange(height) * 14, np.arange(width) * 14
    indices = []
    for i, record in enumerate(records):
        top, left, h, w = record['rectangle_tlhw']
        overlap_y = np.maximum(0, np.minimum(yy + 14, top + h - 14) - np.maximum(yy, top + 14))
        overlap_x = np.maximum(0, np.minimum(xx + 14, left + w - 14) - np.maximum(xx, left + 14))
        mask = overlap_y[:, None] * overlap_x[None, :] >= 157  # ceil(.8 * 196)
        assert mask.any(), ('Empty legacy interior', record)
        indices.append(np.flatnonzero(mask.ravel()) + i * height * width)
    return np.concatenate(indices)


def compile_schedule(sub, cfg, grid_hw=(32, 32), root=ROOT):
    """Return active [steps,512] flat indices and independently check legacy16 SHA.

    An index identifies a genuine image, view and spatial location:
    tuple_id=index//(H*W), image_id=tuple_id//8, illumination_id=tuple_id%4,
    y=(index%(H*W))//W, x=index%W. The two legacy tuple families duplicate
    normal/light features; they do not add a training loss or a synthetic view.
    """
    root = Path(root)
    original = json.loads((root / sub['original_training_file']).read_text())
    assert sha256(root / sub['original_training_file']) == sub['original_training_sha256']
    height, width = map(int, grid_hw)
    assert height > 0 and width > 0
    records = []
    for name in sub['fit_paths']:
        records.extend(rectangle_metadata(name, sub['seed'], cfg['training_conditions'],
                                          (height * 14, width * 14)))
    # Published geometry was square; shape comes from the current actual grid.
    old_fit_records = original['quadruplets'][:len(original['fit_paths']) * 8]
    assert records[:len(old_fit_records)] == old_fit_records, 'Legacy RNG/geometry metadata changed'
    interior = interior_indices(records, (height, width))
    total = len(records) * height * width
    c = cfg['training']
    rng = np.random.default_rng(c['batch_seed_offset'] + sub['seed'])
    active = np.empty((c['steps'], c['effective_center_patches']), dtype=np.int64)
    full_hash, active_hash = hashlib.sha256(), hashlib.sha256()
    for i in range(c['steps']):
        normal = rng.integers(total, size=c['effective_center_patches'])
        inactive = rng.choice(interior, size=c['legacy_batch_size'] - len(normal))
        active[i] = normal
        full_hash.update(np.concatenate((normal, inactive)).astype('<i8').tobytes())
        active_hash.update(normal.astype('<i8').tobytes())
    metadata = {'effective_center_patches': c['effective_center_patches'],
        'grid_hw': [height, width], 'steps': c['steps'], 'normal_tuple_count': len(records),
        'active_schedule_sha256': active_hash.hexdigest(),
        'legacy_full_schedule_sha256': full_hash.hexdigest(),
        'inactive_draws_used_only_for_rng_compatibility': True,
        'legacy16_schedule_verified': False}
    if sub['shots'] == 16:
        expected = original['models'][0]['batch_schedule_sha256']
        assert full_hash.hexdigest() == expected, ('Legacy center schedule differs', sub['category'], sub['seed'])
        metadata['legacy16_schedule_verified'] = True
    return active, metadata


def verify_data(manifest, cfg, root=ROOT):
    """Check every audited image/mask byte, plus selected training memberships."""
    root = Path(root)
    inventory = json.loads((root / cfg['inventory']).read_text())
    data_root = root / cfg['data_root']
    expected = {r['path'] for r in inventory}
    actual = {p.relative_to(data_root).as_posix() for p in data_root.rglob('*.png')}
    assert actual == expected, {'missing': sorted(expected-actual), 'extra': sorted(actual-expected)}
    for record in inventory:
        path = data_root / record['path']
        assert path.stat().st_size == record['bytes'], path
        assert sha256(path) == record['sha256'], path
    assert manifest['inventory_sha256'] == sha256(root / cfg['inventory'])
    return {'status': 'passed', 'checked_pngs': len(inventory),
            'inventory_sha256': manifest['inventory_sha256'], 'exact_png_membership': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-data', action='store_true')
    parser.add_argument('--verify-schedules', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve previous manifests; use a new output path'
    cfg = json.loads(args.config.read_text())
    manifest = prepare(cfg)
    if args.verify_data:
        manifest['data_check'] = verify_data(manifest, cfg)
    if args.verify_schedules:
        manifest['schedules'] = []
        for sub in manifest['subsets']:
            _, record = compile_schedule(sub, cfg)
            manifest['schedules'].append({'category': sub['category'], 'seed': sub['seed'],
                                           'shots': sub['shots'], **record})
    save(args.output, manifest)
    print(f"Prepared {len(manifest['subsets'])} nested subsets; 2520 evaluations, 630 reused, 1890 new; 225 new heads.")


if __name__ == '__main__':
    main()
