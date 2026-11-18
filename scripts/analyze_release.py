"""Recompute saved per-cell tables without data, weights, CUDA or private files.

Pairs are formed within category/seed/condition before equal-weight averaging.
The three seeds share a test set: sample SD is descriptive, not significance.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
DEV = {'bottle','cable','carpet'}
COMPARISONS = {
    'adapter_full_20261102': [('A1','A0')],
    'mvtec_extension_20261103': [('A1','A0')],
    'anchor_ablation_20261102': [('A1-no-anchor','A1'),('A1-no-anchor','A0')],
    'a1_capacity_data_cnn_20261111': [('MLP64-16','MLP32-16'),('MLP32-32','MLP32-16'),
        ('MLP64-32','MLP64-16'),('MLP64-32','MLP32-32'),('CNN1-32','MLP64-32'),
        ('CNN3-32','CNN1-32'),('MLP32-16','A0-16'),('MLP64-16','A0-16'),
        ('MLP32-32','A0-32'),('MLP64-32','A0-32'),('CNN1-32','A0-32'),('CNN3-32','A0-32')],
    'a1_loss_study_20261111': [('L1','A1'),('L2','A1')],
    'a1_expanded_mlp768_20261116': [('MLP768-16','A1')],
    'a1_expanded_mlp768_full_20261116': [('MLP768-16','A1')],
    'defect_adapter_20261102': [('A1','A0'),('A2','A1'),('A3','A1')],
}
EXPECTED = dict(zip(COMPARISONS, [126,630,189,2520,1260,126,630,36]))


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(cells, field):
    seeds=sorted({r['seed'] for r in cells})
    by_seed=[]
    for seed in seeds:
        selected=[r for r in cells if r['seed']==seed]
        cats=sorted({r['category'] for r in selected})
        by_seed.append(stats.mean(stats.mean(r[field] for r in selected if r['category']==cat) for cat in cats))
    return {'mean_pp':100*stats.mean(by_seed),'seed_sample_sd_pp':100*stats.stdev(by_seed) if len(by_seed)>1 else 0.,
            'seed_means_pp':';'.join(format(100*v,'.12g') for v in by_seed)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'output/reanalysis')
    args=p.parse_args();assert not args.out.exists(),'Use a new output directory.'
    args.out.mkdir(parents=True)
    all_summary=[];all_pairs=[];paired_summary=[];classifications=[]
    counts={}
    for study,comparisons in COMPARISONS.items():
        with (ROOT/'results/metrics'/study/'metrics.csv').open(newline='') as f:
            rows=list(csv.DictReader(f))
        assert len(rows)==EXPECTED[study],(study,len(rows))
        for r in rows:
            r['method']=r.get('arm')or r['method'];r['seed']=int(r['seed'])
            for metric in ('pixel_ap','image_auroc'):
                r[metric]=float(r[metric]);assert 0<=r[metric]<=1
        lookup={(r['method'],r['category'],r['seed'],r['condition']):r for r in rows}
        assert len(lookup)==len(rows),'Duplicate method/category/seed/condition'
        methods=sorted({r['method'] for r in rows});cats=sorted({r['category'] for r in rows})
        seeds=sorted({r['seed'] for r in rows});conditions=sorted({r['condition'] for r in rows})
        assert len(rows)==len(methods)*len(cats)*len(seeds)*len(conditions)
        counts[study]={'cells':len(rows),'methods':methods,'categories':len(cats),'seeds':seeds,'conditions':conditions}
        groups={'all':set(cats),'development3':set(cats)&DEV,'extension12':set(cats)-DEV}
        for group,categories in groups.items():
            if not categories:continue
            for scope in ('clean','shifted'):
                selected=[r for r in rows if r['category'] in categories and (r['condition']=='clean')==(scope=='clean')]
                for method in methods:
                    cells=[r for r in selected if r['method']==method]
                    for metric in ('pixel_ap','image_auroc'):
                        all_summary.append({'study':study,'group':group,'scope':scope,'method':method,'metric':metric,**summarize(cells,metric)})
        for candidate,control in comparisons:
            pairs=[]
            for r in rows:
                if r['method']!=candidate:continue
                b=lookup[(control,r['category'],r['seed'],r['condition'])]
                q={'study':study,'candidate':candidate,'control':control,'category':r['category'],'seed':r['seed'],
                   'condition':r['condition'],'pixel_ap_delta':r['pixel_ap']-b['pixel_ap'],
                   'image_auroc_delta':r['image_auroc']-b['image_auroc']}
                pairs.append(q);all_pairs.append(q)
            for group,categories in groups.items():
                if not categories:continue
                for scope in ('clean','shifted'):
                    selected=[r for r in pairs if r['category'] in categories and (r['condition']=='clean')==(scope=='clean')]
                    for metric in ('pixel_ap_delta','image_auroc_delta'):
                        paired_summary.append({'study':study,'group':group,'scope':scope,'candidate':candidate,
                            'control':control,'metric':metric,**summarize(selected,metric)})
            for cat in cats:
                delta=100*stats.mean(r['pixel_ap_delta'] for r in pairs if r['category']==cat and r['condition']!='clean')
                classifications.append({'study':study,'candidate':candidate,'control':control,'category':cat,
                    'shifted_pixel_ap_delta_pp':delta,'classification':'improved' if delta>.1 else 'degraded' if delta<-.1 else 'tied'})
    csv_write(args.out/'method_summary.csv',all_summary)
    csv_write(args.out/'paired_cells.csv',all_pairs)
    csv_write(args.out/'paired_summary.csv',paired_summary)
    csv_write(args.out/'category_classification.csv',classifications)
    (args.out/'matrix_audit.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(f'Checked {sum(c["cells"] for c in counts.values())} archived cells across {len(counts)} studies; overlapping reused controls are not independent runs.')
    print(args.out.resolve())


if __name__=='__main__':
    main()
