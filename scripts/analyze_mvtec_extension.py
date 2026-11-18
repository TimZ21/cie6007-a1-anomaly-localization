"""Audit frozen A1 on twelve new MVTec categories, separating development results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import faiss
import numpy as np
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_mvtec_extension as runner
from analyze_defect_adapter import stats
from src.defect_adapter import batch_indices
from src.evaluation import pixel_average_precision

METHODS = ['A0', 'A1']
DIAGNOSTICS = ['illumination_squared_l2', 'anchor_squared_l2',
               'cross_defect_squared_error', 'cross_to_teacher_mean_norm_ratio',
               'cross_defect_cosine']


def summarize(cfg, metrics, categories=None):
    """Equal-category/condition means per seed; SD of paired changes in pp."""
    categories = cfg['categories'] if categories is None else categories
    conditions = [c['name'] for c in cfg['conditions']]
    assert categories and len(categories) == len(set(categories))
    assert 'clean' in conditions and len(set(conditions)) == len(conditions)
    assert len(cfg['seeds']) >= 2 and len(set(cfg['seeds'])) == len(cfg['seeds'])
    shifts = [c for c in conditions if c != 'clean']; assert shifts
    lookup = {(m['category'], m['seed'], m['method'], m['condition']): m for m in metrics}
    assert len(lookup) == len(metrics), 'Duplicate metric records'
    result = {'categories': list(categories), 'methods': {}, 'paired': {}}
    for method in METHODS:
        result['methods'][method] = {}
        for group, names in [('clean', ['clean']), ('shift', shifts)]:
            for metric in ['pixel_ap', 'image_auroc']:
                values = [100 * float(np.mean([
                    lookup[cat, seed, method, condition][metric]
                    for cat, condition in itertools.product(categories, names)]))
                    for seed in cfg['seeds']]
                result['methods'][method][f'{group}_{metric}'] = stats(values)
    paired = {key: stats(np.array(result['methods']['A1'][key]['values']) -
                         np.array(result['methods']['A0'][key]['values']))
              for key in result['methods']['A0']}
    paired['shift_gain_minus_clean_gain_pixel_ap'] = stats(
        np.array(paired['shift_pixel_ap']['values']) -
        np.array(paired['clean_pixel_ap']['values']))
    result['paired']['A1_minus_A0'] = paired
    return result


def summarize_groups(cfg, metrics):
    new, old = cfg['categories'], cfg['development_categories']
    assert new and old and set(new).isdisjoint(old)
    return {'new12_primary': summarize(cfg, metrics, new),
            'old3_development': summarize(cfg, metrics, old),
            'all15_secondary': summarize(cfg, metrics, new + old)}


def tensor_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def checked_array(path, digest, shape):
    path = Path(path); assert runner.sha256(path) == digest, path
    value = np.load(path, allow_pickle=False)
    assert value.shape == shape and value.dtype == np.float32, (path, value.shape, value.dtype)
    assert np.isfinite(value).all(), path
    return value


@torch.inference_mode()
def audit(out, cfg, rec):
    assert cfg['methods'] == METHODS and cfg['shots'] == [16] and cfg['seeds'] == [0, 1, 2]
    assert len(cfg['categories']) == 12 and cfg['development_categories'] == ['bottle', 'cable', 'carpet']
    assert set(cfg['categories']).isdisjoint(cfg['development_categories'])
    assert len(cfg['conditions']) == 7
    expected = set(itertools.product(cfg['categories'], cfg['seeds'], METHODS,
                                     [c['name'] for c in cfg['conditions']]))
    assert len(expected) == 504
    assert rec['status'] == 'completed' and rec['completed_evaluations'] == 504
    assert rec['completed_training_sets'] == len(rec['training_units']) == 36
    assert rec['fit_finished_utc'] <= rec['evaluation_started_utc']
    ident = runner.identity(cfg)
    assert all(rec[k] == v for k, v in ident.items())
    assert runner.sha256(out / 'config.json') == rec['config_sha256']
    preflight = json.loads((out / 'preflight.json').read_text())
    assert runner.sha256(out / 'preflight.json') == rec['preflight_sha256']
    assert preflight['status'] == 'passed' and preflight['source_commit'] == rec['source_commit']
    assert preflight['identity'] == ident
    assert preflight['data_validation_sha256'] == ident['data_validation_sha256']
    bridges = preflight['old_control_bridge']
    assert len(bridges) == 9
    assert {(b['category'], b['seed']) for b in bridges} == set(itertools.product(cfg['development_categories'], cfg['seeds']))
    bridge_errors = {m: 0. for m in METHODS}
    for bridge in bridges:
        assert bridge['test_path'].startswith(bridge['category'] + '/test/')
        assert set(bridge['methods']) == set(METHODS)
        for method, values in bridge['methods'].items():
            assert set(values) == {c['name'] for c in cfg['conditions']}
            assert all(np.isfinite(v) and 0 <= v < 2e-6 for v in values.values())
            bridge_errors[method] = max(bridge_errors[method], max(values.values()))
    assert json.loads((runner.BASE / 'verification.json').read_text())['status'] == 'passed'
    subsets, inventory = runner.selected(cfg, ident)
    assert len(subsets) == 36
    training, metrics, diagnostic_rows = [], [], []
    seen, image_count, expected_image_count = set(), 0, 0
    errors = {k: 0. for k in ['independent_auroc', 'reconstructed_ap', 'patch_to_image',
                              'bank', 'faiss', 'diagnostics', 'zero_adapter_feature']}
    sklearn_checks = []; model = runner.ad.load(); t = cfg['training']
    with ThreadPoolExecutor(max_workers=4) as workers:
        for cat in cfg['categories']:
            paths, names, labels, masks, shapes = runner.test_data(cat, inventory)
            gpu_masks = torch.from_numpy(masks).cuda()
            expected_image_count += len(paths) * len(cfg['seeds']) * len(METHODS) * len(cfg['conditions'])
            queries = {c['name']: runner.normalized(runner.ad.features(model, runner.rgb(paths[0]), c))
                       for c in cfg['conditions']}
            for sub in [s for s in subsets if s['category'] == cat]:
                seed = sub['seed']; key = f'{cat}_s{seed}_k16'
                trpath = out / 'training' / f'{cat}_s{seed}.json'
                recorded = next(r for r in rec['training_units'] if r['path'] == trpath.as_posix())
                assert runner.sha256(trpath) == recorded['sha256']
                tr, arrays, interior = runner.load_training(cat, seed, out)
                assert tr == json.loads(trpath.read_text()) and tr['status'] == 'fitted'
                training.append(tr)
                fit_names, diagnostic_names = runner.partition(sub, cfg)
                assert tr['fit_paths'] == fit_names and tr['diagnostic_paths'] == diagnostic_names
                assert len(fit_names) == 12 and len(diagnostic_names) == 4
                assert set(fit_names).isdisjoint(diagnostic_names)
                assert tr['reference_paths'] == [r['path'] for r in sub['train']]
                assert all('/train/good/' in name for name in fit_names + diagnostic_names)
                for name in fit_names + diagnostic_names:
                    assert runner.sha256(Path('data/mvtec_ad') / name) == inventory[name]['sha256']
                assert runner.sha256(Path(tr['cache'])) == tr['cache_sha256']
                assert all(v.shape == (128, 1024, 384) and np.isfinite(v).all() for v in arrays.values())
                assert interior.shape == (128, 1024)
                assert [r['path'] for r in tr['quadruplets']] == [n for n in fit_names + diagnostic_names for _ in range(8)]
                assert {r['condition']['name'] for r in tr['quadruplets']} == {c['name'] for c in cfg['training_conditions']}
                rng = np.random.default_rng(t['batch_seed_offset'] + seed); schedule = hashlib.sha256()
                indices = np.flatnonzero(interior[:96].flatten())
                for _ in range(t['steps']):
                    schedule.update(batch_indices(rng, 96 * 1024, indices, t['batch_size']).astype('<i8').tobytes())
                assert len(tr['models']) == 1 and tr['models'][0]['method'] == 'A1'
                info = tr['models'][0]
                assert info['checkpoint_step'] == t['steps'] == 1000
                assert info['batch_schedule_sha256'] == schedule.hexdigest()
                torch.manual_seed(t['initialization_seed_offset'] + seed)
                head = runner.DefectAdapter(bottleneck=t['bottleneck']).cuda().eval()
                assert runner.state_hash(head) == info['initial_state_sha256']
                assert sum(p.numel() for p in head.parameters()) == info['parameters'] == 24992
                initial = head(torch.from_numpy(arrays['normal'][0]).cuda())
                errors['zero_adapter_feature'] = max(errors['zero_adapter_feature'],
                    float((initial - torch.from_numpy(arrays['normal'][0]).cuda()).abs().max()))
                assert runner.sha256(Path(info['checkpoint'])) == info['checkpoint_sha256']
                head.load_state_dict(torch.load(info['checkpoint'], weights_only=True, map_location='cuda'))
                head.requires_grad_(False)
                assert runner.state_hash(head) == info['final_state_sha256']
                for method, current, recorded_diag in [('A0', None, tr['A0_diagnostic']),
                                                       ('A1', head, info['diagnostic'])]:
                    measured = runner.diagnostics(current, {k: v[96:] for k, v in arrays.items()}, interior[96:])
                    assert set(measured) == set(recorded_diag)
                    for name, value in recorded_diag.items():
                        assert np.isfinite(value)
                        errors['diagnostics'] = max(errors['diagnostics'], abs(measured[name] - value))
                    diagnostic_rows.append({'category': cat, 'seed': seed, 'method': method, **measured})
                del arrays, interior, initial
                u = json.loads((out / 'units' / f'{key}.json').read_text())
                assert u['status'] == 'completed' and len(u['metrics']) == 14
                assert u['source_commit'] == rec['source_commit'] and u['config_sha256'] == rec['config_sha256']
                assert u['reference_paths'] == tr['reference_paths']
                assert u['bank_vectors'] == 131072 and u['bank_bytes'] == 201326592
                assert set(u['banks']) == set(METHODS)
                sp = out / 'units' / u['image_scores_file']; assert runner.sha256(sp) == u['image_scores_sha256']
                with sp.open(newline='') as f: rows = list(csv.DictReader(f))
                assert len(rows) == len(paths) * 14; image_count += len(rows)
                raw_info = u['banks']['A0']
                raw = checked_array(raw_info['file'], raw_info['sha256'], (131072, 384))
                assert tensor_sha(raw) == u['raw_bank_sha256']; raw_gpu = torch.from_numpy(raw).cuda()
                for method in METHODS:
                    current = head if method == 'A1' else None
                    bank_info = u['banks'][method]
                    bank = checked_array(bank_info['file'], bank_info['sha256'], (131072, 384))
                    reconstructed = runner.transform(current, raw_gpu).cpu().numpy()
                    errors['bank'] = max(errors['bank'], float(np.max(np.abs(bank - reconstructed))))
                    index = faiss.IndexFlatL2(384); index.add(bank)
                    for c in cfg['conditions']:
                        cname = c['name']; unique = (cat, seed, method, cname)
                        assert unique in expected and unique not in seen; seen.add(unique)
                        m = next(r for r in u['metrics'] if r['method'] == method and r['condition'] == cname)
                        assert m['category'] == cat and m['seed'] == seed and m['shots'] == 16
                        assert m['n_test'] == len(paths) and m['n_anomaly'] == int(labels.sum())
                        assert 0 <= m['image_auroc'] <= 1 and 0 <= m['pixel_ap'] <= 1
                        metrics.append(m)
                        sr = [r for r in rows if r['method'] == method and r['condition'] == cname]
                        assert [r['path'] for r in sr] == names
                        assert [int(r['label']) for r in sr] == labels.astype(int).tolist()
                        scores = np.array([float(r['score']) for r in sr]); assert np.isfinite(scores).all()
                        difference = scores[labels, None] - scores[~labels][None, :]
                        auc = float(((difference > 0) + .5 * (difference == 0)).mean())
                        errors['independent_auroc'] = max(errors['independent_auroc'], abs(auc - m['image_auroc']))
                        patch = checked_array(m['patch_file'], m['patch_sha256'], (len(paths), 1024))
                        errors['patch_to_image'] = max(errors['patch_to_image'],
                            float(np.max(np.abs(scores - [runner.ad.image_score(p) for p in patch]))))
                        maps = np.stack(list(workers.map(runner.ad.map224, patch, shapes)))
                        ap = pixel_average_precision(gpu_masks, torch.from_numpy(maps).cuda())
                        errors['reconstructed_ap'] = max(errors['reconstructed_ap'], abs(ap - m['pixel_ap']))
                        if seed == 0 and cname == 'clean':
                            error = abs(float(average_precision_score(masks.flatten(), maps.flatten())) - ap)
                            sklearn_checks.append({'category': cat, 'method': method, 'error': error})
                        q = runner.transform(current, queries[cname]).cpu().numpy()[::32].copy()
                        independent = index.search(q, 1)[0][:, 0] * .5
                        errors['faiss'] = max(errors['faiss'], float(np.max(np.abs(independent - patch[0, ::32]))))
                        del maps
                    del bank, reconstructed, index
                del head, raw, raw_gpu
                print(f'Audited {key}: 14 evaluations, frozen A1 training, diagnostics, banks and CPU FAISS', flush=True)
            del gpu_masks, queries
            gc.collect(); torch.cuda.empty_cache()
    assert seen == expected and len(metrics) == 504 and image_count == expected_image_count
    for key in ['independent_auroc', 'reconstructed_ap', 'patch_to_image', 'diagnostics']:
        assert errors[key] < 1e-10, (key, errors[key])
    assert errors['bank'] < 1e-6 and errors['faiss'] < 2e-5 and errors['zero_adapter_feature'] < 1e-6
    assert len(sklearn_checks) == 24 and max(r['error'] for r in sklearn_checks) < 1e-12
    # Old development categories remain pinned, separately audited reused results.
    for cat, seed in itertools.product(cfg['development_categories'], cfg['seeds']):
        unit = json.loads((runner.BASE / 'units' / f'{cat}_s{seed}_k16.json').read_text())
        assert unit['status'] == 'completed' and len(unit['metrics']) == 14
        metrics.extend(unit['metrics'])
        tr = json.loads((runner.BASE / 'training' / f'{cat}_s{seed}.json').read_text())
        for method, values in [('A0', tr['A0_diagnostic']), ('A1', tr['models'][0]['diagnostic'])]:
            diagnostic_rows.append({'category': cat, 'seed': seed, 'method': method, **values})
    assert len(metrics) == 630 and len(diagnostic_rows) == 90
    verification = {'status': 'passed', 'utc': runner.utc(), 'new_evaluations': 504,
        'new_image_rows': image_count, 'reused_evaluations': 126, 'combined_evaluations': 630,
        'reused_audit_sha256': runner.sha256(runner.BASE / 'verification.json'),
        'preflight_sha256': runner.sha256(out / 'preflight.json'),
        'old_control_bridge_max_error': bridge_errors,
        'max_errors': errors, 'independent_sklearn_clean_seed0': sklearn_checks,
        'source_config_inventory_split_cache_checkpoint_bank_scores_complete_grid': 'passed',
        'unchanged_A1_initialization_batches_partition_final_step_and_diagnostics': 'passed',
        'all_36_new_heads_fitted_before_new_category_evaluation': True,
        'old_development_categories_reused_separately': True}
    return training, metrics, diagnostic_rows, verification


def write_csv(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]), extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def report(out, cfg, rec, training, metrics, diagnostic_rows, verification):
    result = summarize_groups(cfg, metrics)
    categories = cfg['categories'] + cfg['development_categories']
    result['by_category'] = {cat: summarize(cfg, metrics, [cat]) for cat in categories}
    lookup = {(m['category'], m['seed'], m['method'], m['condition']): m for m in metrics}
    pairs = []
    for cat, seed, c in itertools.product(categories, cfg['seeds'], cfg['conditions']):
        a, b = [lookup[cat, seed, m, c['name']] for m in METHODS]
        pairs.append({'category': cat, 'seed': seed, 'condition': c['name'],
            'scope': 'new12_primary' if cat in cfg['categories'] else 'old3_development',
            **{f'{metric}_delta_pp': 100 * (b[metric] - a[metric]) for metric in ['pixel_ap', 'image_auroc']}})
    result['condition_pairs'] = pairs
    result['category_seed_pairs'] = [{'category': cat, 'seed': seed,
        'scope': 'new12_primary' if cat in cfg['categories'] else 'old3_development',
        **{f'{group}_{metric}_delta_pp': float(np.mean([p[f'{metric}_delta_pp'] for p in pairs
            if p['category'] == cat and p['seed'] == seed and ((p['condition'] == 'clean') == (group == 'clean'))]))
           for group, metric in itertools.product(['clean', 'shift'], ['pixel_ap', 'image_auroc'])}}
        for cat, seed in itertools.product(categories, cfg['seeds'])]
    for name in ['new12_primary', 'old3_development', 'all15_secondary']:
        cats = result[name]['categories']; cp = [p for p in pairs if p['category'] in cats]
        sp = [p for p in cp if p['condition'] != 'clean']
        cs = [p for p in result['category_seed_pairs'] if p['category'] in cats]
        result[name]['stability'] = {
            'positive_shift_category_means': sum(result['by_category'][cat]['paired']['A1_minus_A0']['shift_pixel_ap']['mean'] > 0 for cat in cats),
            'total_categories': len(cats), 'positive_shift_category_seed_means': sum(p['shift_pixel_ap_delta_pp'] > 0 for p in cs),
            'total_category_seed_pairs': len(cs), 'positive_shift_condition_pairs': sum(p['pixel_ap_delta_pp'] > 0 for p in sp),
            'total_shift_condition_pairs': len(sp), 'worst_shift_condition_pair': min(sp, key=lambda p: p['pixel_ap_delta_pp'])}
        result[name]['regressions'] = {metric: [p for p in cp if p[f'{metric}_delta_pp'] < 0]
                                       for metric in ['pixel_ap', 'image_auroc']}
    result['diagnostics'] = {cat: {method: {key: stats([r[key] for r in diagnostic_rows
        if r['category'] == cat and r['method'] == method]) for key in DIAGNOSTICS}
        for method in METHODS} for cat in categories}
    result['new12_costs'] = {method: {'head_fit_seconds': [0. if method == 'A0' else tr['models'][0]['fit_s'] for tr in training],
        'scoring_ms_per_image': 1000 * sum(m['scoring_s'] for m in metrics if m['category'] in cfg['categories'] and m['method'] == method)
        / sum(m['n_test'] for m in metrics if m['category'] in cfg['categories'] and m['method'] == method)} for method in METHODS}
    result['interpretation'] = {'primary': 'Twelve categories not used in the local A1/R1 development rounds; settings frozen before their evaluation.',
        'development': 'Three previously inspected categories are reused and reported separately.',
        'secondary': 'All fifteen categories are descriptive; do not substitute this combined result for the primary twelve-category result.',
        'limitations': 'One benchmark and three reference-sampling seeds; common test images and conditions are correlated. No SOTA claim, significance claim, test-label tuning, checkpoint reselection or outcome-based category omission.'}
    runner.save(out / 'summary.json', result); runner.save(out / 'verification.json', verification)
    write_csv(out / 'metrics.csv', metrics, ['category', 'seed', 'shots', 'method', 'condition', 'image_auroc', 'pixel_ap', 'n_test', 'n_anomaly', 'scoring_s'])
    write_csv(out / 'paired_conditions.csv', pairs)
    write_csv(out / 'category_seed_pairs.csv', result['category_seed_pairs'])
    write_csv(out / 'diagnostics.csv', diagnostic_rows)
    keys = ['clean_pixel_ap', 'shift_pixel_ap', 'clean_image_auroc', 'shift_image_auroc']
    lines = ['# Frozen A1: twelve-category MVTec extension', '',
        f"Frozen source: `{rec['source_commit']}`. All 504 new evaluations / {verification['new_image_rows']:,} new image scores audited; 126 audited development evaluations reused.",
        'No method, loss weight, rank, checkpoint rule, training schedule or evaluation setting was selected from these new test outcomes.', '',
        'The primary analysis is the twelve new categories. The old three development categories and all fifteen combined are separate secondary context.', '',
        '| Scope | Method | Clean pixel AP | Six-shift pixel AP | Clean I-AUROC | Six-shift I-AUROC |',
        '|---|---|---:|---:|---:|---:|']
    for name in ['new12_primary', 'old3_development', 'all15_secondary']:
        for method in METHODS:
            lines.append('| ' + name + ' | ' + method + ' | ' + ' | '.join(
                f"{result[name]['methods'][method][key]['mean']:.3f} +/- {result[name]['methods'][method][key]['sample_sd']:.3f}" for key in keys) + ' |')
        paired = result[name]['paired']['A1_minus_A0']
        lines.append('| ' + name + ' | A1 minus A0 (pp) | ' + ' | '.join(
            f"{paired[key]['mean']:+.3f} +/- {paired[key]['sample_sd']:.3f}" for key in keys) + ' |')
    lines += ['', 'Each category and condition receives equal weight within each seed. SD is sample SD across three paired reference-sampling seeds, not independent dataset replication. The combined fifteen-category mean weights categories 12:3, not the two scope means 1:1.', '',
        '| Category | Scope | A0 shift AP | A1 shift AP | Shift gain (pp) | Clean gain (pp) |',
        '|---|---|---:|---:|---:|---:|']
    for cat in categories:
        d = result['by_category'][cat]; p = d['paired']['A1_minus_A0']
        scope = 'new' if cat in cfg['categories'] else 'development'
        lines.append(f"| {cat} | {scope} | {d['methods']['A0']['shift_pixel_ap']['mean']:.3f} | {d['methods']['A1']['shift_pixel_ap']['mean']:.3f} | {p['shift_pixel_ap']['mean']:+.3f} +/- {p['shift_pixel_ap']['sample_sd']:.3f} | {p['clean_pixel_ap']['mean']:+.3f} +/- {p['clean_pixel_ap']['sample_sd']:.3f} |")
    lines += ['', 'All 315 category/seed/condition paired changes are in paired_conditions.csv; all 45 category/seed means are in category_seed_pairs.csv. Every regression remains in summary.json and these tables.', '',
        '## Paired stability and robustness interpretation', '']
    for name in ['new12_primary', 'old3_development', 'all15_secondary']:
        st = result[name]['stability']; p = result[name]['paired']['A1_minus_A0']['shift_gain_minus_clean_gain_pixel_ap']
        lines.append(f"- {name}: positive shift-AP category means {st['positive_shift_category_means']}/{st['total_categories']}; category/seed means {st['positive_shift_category_seed_means']}/{st['total_category_seed_pairs']}; condition pairs {st['positive_shift_condition_pairs']}/{st['total_shift_condition_pairs']}. Shift gain minus clean gain: {p['mean']:+.3f} +/- {p['sample_sd']:.3f} pp.")
    lines += ['', 'Sign counts are descriptive. Improvement on both clean and shifted images alone does not demonstrate a shift-specific robustness advantage.', '',
        '## Normal-only diagnostics', '',
        'Four normal images per category/seed are excluded from fitting and retained in the reference bank. Their training-family photometric and synthetic views diagnose behavior; synthetic differences do not certify preservation of real defects. No diagnostic selected checkpoints or changed settings.', '',
        '| Category | Method | Normal light squared L2 | Anchor squared L2 | Synthetic cross-difference error | Difference norm ratio |',
        '|---|---|---:|---:|---:|---:|']
    for cat, method in itertools.product(categories, METHODS):
        d = result['diagnostics'][cat][method]
        lines.append('| ' + cat + ' | ' + method + ' | ' + ' | '.join(f"{d[k]['mean']:.6f}" for k in DIAGNOSTICS[:4]) + ' |')
    lines += ['', 'Per-seed diagnostics, including directional cosine, are in diagnostics.csv.', '',
        '## Audit and resources', '',
        'The original DefectAdapter and fit_one(A1) are reused without modification. Audits reconstruct initialization and every sampled batch, verify normal-only 12/4 partitions, final step 1000, frozen artifact hashes, A1 banks and diagnostics. Every new AP is recomputed from saved patches, every AUROC independently pair-counted; clean seed-0 AP is checked with sklearn and 32 first-image query patches per condition with CPU FAISS.',
        'The pre-training old-category bridge is stored in preflight.json. Reused development metrics retain their original audit and hashes.',
        'Audit maximum errors: `' + json.dumps(verification['max_errors'], sort_keys=True) + '`.', '',
        f"New normal cache extraction and fitting: {rec['fit_elapsed_s']/60:.2f} min; evaluation: {rec['evaluation_elapsed_s']/60:.2f} min; audit: {verification['audit_elapsed_s']/60:.2f} min.",
        'Resource counters: `' + json.dumps({k: rec[k] for k in ['fit_resources', 'eval_resources']}, sort_keys=True) + '`.', '',
        'This experiment tests transfer of the frozen A1 recipe within MVTec AD. It does not establish cross-dataset generalization, statistical independence of seeds, or SOTA. All outcomes are retained; no gate triggers tuning on these test results.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({name: {'paired': result[name]['paired'], 'stability': result[name]['stability']}
        for name in ['new12_primary', 'old3_development', 'all15_secondary']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', default='mvtec_extension_20261103')
    args = parser.parse_args(); runner.setup(); out = Path('results/metrics') / args.run_id
    cfg = json.loads((out / 'config.json').read_text()); rec = json.loads((out / 'run.json').read_text())
    start = time.perf_counter(); training, metrics, diagnostic_rows, verification = audit(out, cfg, rec)
    verification['audit_elapsed_s'] = time.perf_counter() - start
    report(out, cfg, rec, training, metrics, diagnostic_rows, verification)


if __name__ == '__main__':
    main()
