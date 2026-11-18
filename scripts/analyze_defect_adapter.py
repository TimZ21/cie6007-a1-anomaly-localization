"""Audit the complete fixed A0-A3 pilot, then report every paired result."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import itertools
import json
from pathlib import Path
import sys
import time

import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_assets import sha256
from run_e1 import save, utc
from run_e2 import normalized
from run_backbone_comparison import setup, test_data, rgb
from run_defect_adapter import identity, selected, partition, state_hash, transform
from src import anomalydino as ad
from src.defect_adapter import DefectAdapter, batch_indices
from src.evaluation import pixel_average_precision


def stats(values):
    x = np.asarray(values, dtype=float)
    return {'mean': float(x.mean()), 'sample_sd': float(x.std(ddof=1)),
            'values': x.tolist(), 'positive_seeds': int((x > 0).sum())}


@torch.inference_mode()
def audit(out, cfg, rec):
    import hashlib
    assert rec['status'] == 'completed' and rec['completed_evaluations'] == 36
    assert rec['fit_finished_utc'] <= rec['evaluation_started_utc']
    assert all(rec[k] == v for k, v in identity(cfg).items())
    assert sha256(out / 'config.json') == rec['config_sha256']
    subsets, inv = selected(cfg, rec)
    paths, names, labels, masks, shapes = test_data('cable', inv); gpu_masks = torch.from_numpy(masks).cuda()
    model = ad.load()
    sample_queries = {c['name']: normalized(ad.features(model, rgb(paths[0]), c)) for c in cfg['conditions']}
    expected = set(itertools.product(cfg['seeds'], cfg['methods'], [c['name'] for c in cfg['conditions']]))
    seen = set(); allmetrics = []; training = []; units = []; image_rows = 0
    errors = {'independent_auroc': 0., 'reconstructed_pixel_ap': 0., 'patch_to_image': 0.,
              'adapter_bank': 0., 'faiss_sample_patch_distance': 0., 'zero_adapter_feature': 0.}
    sklearn = []; t = cfg['training']
    with ThreadPoolExecutor(max_workers=4) as workers:
        for sub in subsets:
            seed = sub['seed']; key = f'cable_s{seed}_k16'
            tr = json.loads((out / 'training' / f'cable_s{seed}.json').read_text()); training.append(tr)
            recorded = next(r for r in rec['training_units'] if r['path'].endswith(f'cable_s{seed}.json'))
            assert sha256(Path(recorded['path'])) == recorded['sha256'] and tr['status'] == 'fitted'
            fit_names, diagnostic_names = partition(sub, cfg)
            assert tr['fit_paths'] == fit_names and tr['diagnostic_paths'] == diagnostic_names
            assert set(fit_names).isdisjoint(diagnostic_names) and len(fit_names) == 12 and len(diagnostic_names) == 4
            assert tr['reference_paths'] == [r['path'] for r in sub['train']]
            assert all('/train/good/' in n for n in fit_names + diagnostic_names)
            assert sha256(Path(tr['cache'])) == tr['cache_sha256']
            with np.load(tr['cache'], allow_pickle=False) as cache:
                interior = np.flatnonzero(cache['interior'][:96].flatten())
            rng = np.random.default_rng(t['batch_seed_offset'] + seed); schedule = hashlib.sha256()
            for _ in range(t['steps']):
                schedule.update(batch_indices(rng, 96 * 1024, interior, t['batch_size']).astype('<i8').tobytes())
            assert len(tr['quadruplets']) == 128
            assert [r['path'] for r in tr['quadruplets']] == [n for n in fit_names + diagnostic_names for _ in range(8)]
            assert {r['condition']['name'] for r in tr['quadruplets']} == {c['name'] for c in cfg['training_conditions']}
            torch.manual_seed(t['initialization_seed_offset'] + seed); initial = DefectAdapter(bottleneck=t['bottleneck']).cuda()
            for m in tr['models']:
                assert m['initial_state_sha256'] == state_hash(initial)
                assert m['batch_schedule_sha256'] == schedule.hexdigest() and m['checkpoint_step'] == t['steps']
                assert sha256(Path(m['checkpoint'])) == m['checkpoint_sha256']
            assert {m['method'] for m in tr['models']} == {'A1', 'A2', 'A3'}
            up = out / 'units' / (key + '.json'); u = json.loads(up.read_text()); units.append(u)
            assert u['status'] == 'completed' and len(u['metrics']) == 12
            assert u['source_commit'] == rec['source_commit'] and u['config_sha256'] == rec['config_sha256']
            assert u['reference_paths'] == tr['reference_paths'] and u['bank_vectors'] == 131072 and u['bank_bytes'] == 201326592
            sp = up.with_name(u['image_scores_file']); assert sha256(sp) == u['image_scores_sha256']
            with sp.open(newline='') as f:
                rows = list(csv.DictReader(f))
            image_rows += len(rows)
            raw_np = np.load(u['banks']['A0']['file'], allow_pickle=False)
            assert hashlib.sha256(raw_np.tobytes()).hexdigest() == u['raw_bank_sha256']
            raw = torch.from_numpy(raw_np).cuda()
            errors['zero_adapter_feature'] = max(errors['zero_adapter_feature'], float((initial(raw[:1024]) - raw[:1024]).abs().max()))
            for method in cfg['methods']:
                b = u['banks'][method]; assert sha256(Path(b['file'])) == b['sha256']
                bank = np.load(b['file'], allow_pickle=False); assert bank.shape == (131072, 384)
                head = None
                if method != 'A0':
                    info = next(m for m in tr['models'] if m['method'] == method)
                    head = DefectAdapter(bottleneck=t['bottleneck']).cuda()
                    head.load_state_dict(torch.load(info['checkpoint'], weights_only=True, map_location='cuda'))
                    head.eval(); assert state_hash(head) == info['final_state_sha256']
                expected_bank = transform(head, raw).cpu().numpy()
                errors['adapter_bank'] = max(errors['adapter_bank'], float(np.max(np.abs(bank - expected_bank))))
                index = faiss.IndexFlatL2(384); index.add(bank)
                for c in cfg['conditions']:
                    cname = c['name']; k = (seed, method, cname)
                    assert k in expected and k not in seen; seen.add(k)
                    m = next(r for r in u['metrics'] if r['method'] == method and r['condition'] == cname)
                    allmetrics.append(m)
                    sr = [r for r in rows if r['method'] == method and r['condition'] == cname]
                    assert [r['path'] for r in sr] == names and [int(r['label']) for r in sr] == labels.astype(int).tolist()
                    scores = np.array([float(r['score']) for r in sr]); assert len(scores) == m['n_test']
                    diff = scores[labels, None] - scores[~labels][None, :]
                    auc = float(((diff > 0) + .5 * (diff == 0)).mean())
                    errors['independent_auroc'] = max(errors['independent_auroc'], abs(auc - m['image_auroc']))
                    pp = Path(m['patch_file']); assert sha256(pp) == m['patch_sha256']
                    patch = np.load(pp, allow_pickle=False); assert patch.shape == (len(paths), 1024) and np.isfinite(patch).all()
                    errors['patch_to_image'] = max(errors['patch_to_image'], float(np.max(np.abs(scores - np.array([ad.image_score(p) for p in patch])))))
                    maps = np.stack(list(workers.map(ad.map224, patch, shapes)))
                    ap = pixel_average_precision(gpu_masks, torch.from_numpy(maps).cuda())
                    errors['reconstructed_pixel_ap'] = max(errors['reconstructed_pixel_ap'], abs(ap - m['pixel_ap']))
                    if seed == 0 and cname == 'clean':
                        sk = float(average_precision_score(masks.flatten(), maps.flatten()))
                        sklearn.append({'method': method, 'absolute_error': abs(sk - ap)})
                    q = transform(head, sample_queries[cname]).cpu().numpy()[::32].copy()
                    independently = index.search(q, 1)[0][:, 0] * .5
                    errors['faiss_sample_patch_distance'] = max(errors['faiss_sample_patch_distance'], float(np.max(np.abs(independently - patch[0, ::32]))))
                    del maps
                del head, index, expected_bank, bank
            del raw, raw_np, initial
            print(f'Audited {key}: initialization, minibatches, banks, all metrics and independent FAISS samples', flush=True)
    assert seen == expected and image_rows == len(paths) * 36
    for name in ['independent_auroc', 'reconstructed_pixel_ap', 'patch_to_image']:
        assert errors[name] < 1e-12, (name, errors[name])
    assert errors['adapter_bank'] < 1e-6 and errors['faiss_sample_patch_distance'] < 2e-5 and errors['zero_adapter_feature'] < 1e-6
    assert len(sklearn) == 4 and max(r['absolute_error'] for r in sklearn) < 1e-12
    verification = {'status': 'passed', 'utc': utc(), 'evaluations': 36, 'image_rows': image_rows,
                    'max_errors': errors, 'independent_sklearn_clean_seed0': sklearn,
                    'source_config_split_inventory_weights_cache_checkpoint_and_result_hashes': 'passed',
                    'same_initialization_minibatches_steps_reference_budget': 'passed',
                    'fitting_completed_before_test_evaluation': True,
                    'previous_e2_reproduction': {k: max(u['previous_e2_reproduction'][k] for u in units) for k in units[0]['previous_e2_reproduction']}}
    return training, allmetrics, verification


def summarize(cfg, training, metrics):
    s = {'methods': {}, 'paired': {}, 'diagnostics': {}, 'condition_means': []}
    for method in cfg['methods']:
        s['methods'][method] = {}
        for group, conditions in [('clean', ['clean']), ('shift', ['brightness_070', 'gamma_140'])]:
            for metric in ['image_auroc', 'pixel_ap']:
                values = [100 * np.mean([m[metric] for m in metrics if m['seed'] == seed and m['method'] == method and m['condition'] in conditions]) for seed in cfg['seeds']]
                s['methods'][method][group + '_' + metric] = stats(values)
        diag = [tr['A0_diagnostic'] if method == 'A0' else next(m for m in tr['models'] if m['method'] == method)['diagnostic'] for tr in training]
        s['diagnostics'][method] = {k: stats([d[k] for d in diag]) for k in diag[0] if k != 'interior_patches'}
        selected_metrics = [m for m in metrics if m['method'] == method]
        s['methods'][method]['scoring_ms_per_image'] = 1000 * sum(m['scoring_s'] for m in selected_metrics) / sum(m['n_test'] for m in selected_metrics)
        s['methods'][method]['fit_s_per_seed'] = [0. if method == 'A0' else next(m for m in tr['models'] if m['method'] == method)['fit_s'] for tr in training]
    for method, baseline in [('A1', 'A0'), ('A2', 'A0'), ('A3', 'A0'), ('A3', 'A1'), ('A3', 'A2')]:
        s['paired'][method + '_minus_' + baseline] = {
            k: stats(np.array(s['methods'][method][k]['values']) - s['methods'][baseline][k]['values'])
            for k in ['clean_image_auroc', 'shift_image_auroc', 'clean_pixel_ap', 'shift_pixel_ap']}
    for c, method in itertools.product(cfg['conditions'], cfg['methods']):
        group = [m for m in metrics if m['condition'] == c['name'] and m['method'] == method]
        s['condition_means'].append({'condition': c['name'], 'method': method,
            **{k: stats([m[k] * 100 for m in group]) for k in ['image_auroc', 'pixel_ap']}})
    p, gate = s['paired']['A3_minus_A0'], cfg['continuation_target']
    s['practical_baseline_gate_met'] = bool(p['shift_pixel_ap']['mean'] >= gate['shift_pixel_ap_min_pp']
        and p['clean_pixel_ap']['mean'] >= gate['clean_pixel_ap_min_pp']
        and p['clean_image_auroc']['mean'] >= gate['clean_image_auroc_min_pp']
        and p['shift_pixel_ap']['positive_seeds'] >= gate['min_positive_seeds'])
    s['cross_light_ablation_mean_positive'] = s['paired']['A3_minus_A2']['shift_pixel_ap']['mean'] > 0
    s['continuation_supported'] = s['practical_baseline_gate_met'] and s['cross_light_ablation_mean_positive']
    return s


def report(out, cfg, rec, training, metrics, verified, s):
    save(out / 'verification.json', verified); save(out / 'summary.json', s)
    fields = ['category', 'seed', 'shots', 'method', 'condition', 'image_auroc', 'pixel_ap', 'n_test', 'n_anomaly', 'scoring_s', 'adapter_s', 'mapping_s']
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n'); w.writeheader(); w.writerows(metrics)
    keys = ['clean_image_auroc', 'shift_image_auroc', 'clean_pixel_ap', 'shift_pixel_ap']
    lines = ['# Defect-difference adapter: audited exploratory pilot', '',
        '**Cable only; 16 references; three paired seeds; clean/brightness-0.70/gamma-1.40. No untouched confirmation or SOTA claim.**', '',
        f"Frozen source: `{rec['source_commit']}`. Original main: `{rec['base_commit']}`; previous experimental branch: `{rec['parent_commit']}`.",
        f"All {verified['evaluations']} evaluations / {verified['image_rows']} image scores passed audit. Values are percent, mean +/- sample SD over the three reference/training seeds. These are not independent test datasets.", '',
        '| Method | Clean I-AUROC | Shift I-AUROC | Clean pixel AP | Shift pixel AP |', '|---|---:|---:|---:|---:|']
    for method in cfg['methods']:
        lines.append('| ' + method + ' | ' + ' | '.join(f"{s['methods'][method][k]['mean']:.3f} +/- {s['methods'][method][k]['sample_sd']:.3f}" for k in keys) + ' |')
    lines += ['', '## Paired differences (percentage points)', '', '| Comparison | Clean I-AUROC | Shift I-AUROC | Clean pixel AP | Shift pixel AP |', '|---|---:|---:|---:|---:|']
    for name, values in s['paired'].items():
        lines.append('| ' + name + ' | ' + ' | '.join(f"{values[k]['mean']:+.3f} +/- {values[k]['sample_sd']:.3f}" for k in keys) + ' |')
    for name in ['A3_minus_A0', 'A3_minus_A2']:
        d = s['paired'][name]['shift_pixel_ap']
        lines.append(f"\n{name} shifted-AP seed deltas: " + ', '.join(f'{v:+.3f}' for v in d['values']) + f" pp; {d['positive_seeds']}/3 positive.")
    lines += ['', f"Frozen practical baseline gate met: **{s['practical_baseline_gate_met']}**. A3-minus-A2 mean shifted AP positive: **{s['cross_light_ablation_mean_positive']}**. Continuation supported by both: **{s['continuation_supported']}**.",
        'Gate: A3-A0 shift AP >= +1 pp; clean AP >= -0.5 pp; clean image AUROC >= -0.2 pp; at least two positive shifted-AP seeds. Cross-light attribution also requires A3 to improve over A2. This is a practical decision rule, not statistical significance. No winning ablation is substituted for A3.', '',
        '## Condition breakdown', '', '| Condition | Method | I-AUROC | Pixel AP |', '|---|---|---:|---:|']
    for r in s['condition_means']:
        lines.append(f"| {r['condition']} | {r['method']} | {r['image_auroc']['mean']:.3f} | {r['pixel_ap']['mean']:.3f} |")
    lines += ['', '## Synthetic diagnostics on four normal images excluded from adapter fitting', '',
        '| Method | Normal-light squared L2 | Cross-defect squared error | Cross/teacher mean defect norm | Cross-defect cosine |', '|---|---:|---:|---:|---:|']
    for method, d in s['diagnostics'].items():
        lines.append('| ' + method + ' | ' + ' | '.join(f"{d[k]['mean']:.6f}" for k in ['illumination_squared_l2', 'cross_defect_squared_error', 'cross_to_teacher_mean_norm_ratio', 'cross_defect_cosine']) + ' |')
    lines += ['', 'Diagnostics use eight synthetic quadruplets per held-out normal image and never select hyperparameters/checkpoints. The four images ARE part of the 16-image deployed normal bank. Synthetic fidelity is not real-defect detection accuracy; DINO attention can spread a local image perturbation globally.', '',
        '## Fairness, cost and audit', '',
        'A0: unchanged detector, no adapter training. A1: illumination + anchor. A2: A1 + clean-light teacher defect-difference preservation. A3: A1 + cross-light teacher defect-difference preservation. All trained heads have 24,992 parameters, the same initialization, synthetic cache, sampled minibatches, optimizer and 1,000 fixed steps. No early stopping, sweep, refit or test-dependent model selection.',
        'Each seed uses the same existing 16 normal references and all eight rotations, with 131,072 x 384 float32 bank vectors (192 MiB). The adapters fit 12 of those images; the remaining four are synthetic diagnostic images only. Synthetic defects are never added to the bank. Both bank and query features use the same saved head.', '',
        '| Method | Mean head-fit seconds | Adapter + search ms/image |', '|---|---:|---:|']
    for method, d in s['methods'].items():
        lines.append(f"| {method} | {np.mean(d['fit_s_per_seed']):.2f} | {d['scoring_ms_per_image']:.3f} |")
    lines += ['', f"Normal-only extraction/fitting phase: {rec['fit_elapsed_s']/60:.2f} min. Evaluation phase: {rec['evaluation_elapsed_s']/60:.2f} min. Combined: {rec['total_fit_and_evaluation_s']/60:.2f} min, excluding tests and audit.",
        'All adapter methods share the frozen teacher feature cache; A0 does not require its synthetic extraction/training cost. Deployment heads add approximately 0.095 MiB of float32 parameters to the unchanged bank size. Timings are descriptive sequential laptop measurements, not a deployment speed benchmark. Scoring includes query adaptation and exact search, excludes encoder/transfer/maps/metrics; bank adaptation and cache extraction are separately recorded.',
        'Resource peaks by separate process: `' + json.dumps({k: rec[k] for k in ['fit_resources', 'eval_resources']}, sort_keys=True) + '`.',
        'All metric records were reconstructed from saved patch arrays; image AUROC independently checked by pair counting; clean seed-0 AP independently checked with sklearn for all four methods. Saved banks were reconstructed from raw features and each checkpoint. Independent CPU FAISS checked 32 patches of the first test image per condition/method/seed. Source/data/cache/checkpoint/score hashes and same initialization/minibatch schedules passed.',
        'Previous E2 A0 reproduction errors: `' + json.dumps(verified['previous_e2_reproduction'], sort_keys=True) + '`.',
        'No per-image min-max normalization, threshold fitting, test-batch statistics or test-time adaptation. Pixel AP includes normal and defective images on the common 224px grid. Full checkpoints, caches, banks and patch arrays remain in ignored runs/ on Windows, with hashes in published records.', '',
        f'![Pilot](../../figures/{out.name}.png)', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#476582', '#8A9AA9', '#D4A35F', '#B65151']
    for seed in range(3):
        axes[0].plot(range(4), [s['methods'][m]['shift_pixel_ap']['values'][seed] for m in cfg['methods']], marker='o', label=f'Seed {seed}', alpha=.8)
    axes[0].set_xticks(range(4), cfg['methods']); axes[0].set_ylabel('Pixel AP (%)'); axes[0].set_title('Cable: mean of two lighting shifts'); axes[0].legend()
    errors = [s['diagnostics'][m]['cross_defect_squared_error']['mean'] for m in cfg['methods']]
    axes[1].bar(range(4), np.array(errors) / errors[0], color=colors)
    axes[1].set_xticks(range(4), cfg['methods']); axes[1].set_ylabel('Error relative to A0 (lower is better)'); axes[1].set_title('Held-normal synthetic defect difference')
    for ax in axes:
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Fixed exploratory adapter pilot; synthetic improvement is not detection improvement', fontsize=10)
    fig.tight_layout(); dest = Path('results/figures'); dest.mkdir(exist_ok=True)
    fig.savefig(dest / f'{out.name}.png', dpi=180); fig.savefig(dest / f'{out.name}.pdf'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run-id', default='defect_adapter_20261102'); args = p.parse_args()
    setup(); out = Path('results/metrics') / args.run_id
    cfg = json.loads((out / 'config.json').read_text()); rec = json.loads((out / 'run.json').read_text())
    start = time.perf_counter(); training, metrics, verified = audit(out, cfg, rec)
    verified['audit_elapsed_s'] = time.perf_counter() - start
    s = summarize(cfg, training, metrics); report(out, cfg, rec, training, metrics, verified, s)
    print(json.dumps({'paired': s['paired'], 'continuation_supported': s['continuation_supported'], 'verification': verified}, indent=2), flush=True)


if __name__ == '__main__':
    main()
