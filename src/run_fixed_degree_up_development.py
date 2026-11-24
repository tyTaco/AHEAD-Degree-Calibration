"""Fixed 1.1 low-degree multiplier on four already-explored D intervals.

The fixed candidate uses no labels, but its direction was considered after
observing these datasets. This is retrospective exploration, not independent
confirmation. No training, model inference, or old Test evaluation is done.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from run_label_gate_development import _verify_run
from train_research import file_sha256

PRIOR = '5baa1257a60b6d3a67af8190b56e3876d2bcb85b98e2b48dad1eb7c3b3c20f2e'
METHODS = ('none', 'original', 'continuous', 'fixed_1_1', 'c_pairwise', 'c_guarded')
METRICS = ('AUROC', 'AP', 'PR_AUC')


def metrics(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {'AUROC': float(roc_auc_score(labels, scores)), 'AP': float(average_precision_score(labels, scores)),
            'PR_AUC': float(auc(recall, precision))}


def analyze(project, dataset, previous):
    run_id = previous['run_id']
    root, provenance, runs = _verify_run(project, run_id)
    if provenance['dataset'] != dataset or sorted(x['seed'] for x in runs) != list(range(5)):
        raise ValueError('Unexpected dataset or seeds')
    dev = root / 'degree_development'
    pp, sp = dev / 'development_started.json', dev / 'development_complete.json'
    if any(x.is_symlink() for x in (dev, pp, sp)):
        raise ValueError('Redirected artifacts')
    protocol, seal = json.loads(pp.read_text()), json.loads(sp.read_text())
    if (file_sha256(pp) != seal['protocol_sha256'] or file_sha256(sp) != previous['development_seal_sha256']
            or protocol['dataset'] != dataset or protocol['run_id'] != run_id):
        raise ValueError('Development seal mismatch')
    lo, hi = protocol['segments']['development']
    if [lo, hi] != previous['development_bounds'] or not 0 <= lo < hi <= protocol['segments']['old_test'][0]:
        raise ValueError('Development interval crosses old Test')
    values, expected_degree, expected_times = {}, None, None
    for policy in ('val_auroc', 'train_loss'):
        values[policy] = []
        prior_items = {x['seed']: x for x in previous['results'][policy]}
        if set(prior_items) != set(range(5)):
            raise ValueError('Missing prior seed results')
        for run in sorted(runs, key=lambda x: x['seed']):
            seed = run['seed']; prior = prior_items[seed]
            name = f'{policy}_seed{seed}_scores.npz'; path = dev / name
            if (path.is_symlink() or file_sha256(path) != prior['score_sha256']
                    or seal['artifact_sha256'][name] != prior['score_sha256']
                    or run['policies'][policy]['checkpoint_sha256'] != prior['checkpoint_sha256']):
                raise ValueError('Score or checkpoint hash mismatch')
            with np.load(path, allow_pickle=False) as z:
                raw = z['none'].astype(np.float64); degree = z['development_degrees']
                if (len(raw) != hi-lo or not np.array_equal(z['original'], raw * np.where(degree < 3, .9, 1.))
                        or not np.array_equal(z['continuous'], raw * (1. - .1 * np.exp(-degree / 3.)))):
                    raise ValueError('Baseline score mismatch')
                if expected_degree is None:
                    expected_degree, expected_times = degree.copy(), z['development_times'].copy()
                elif not np.array_equal(degree, expected_degree) or not np.array_equal(z['development_times'], expected_times):
                    raise ValueError('Event axes differ across runs')
                fixed = raw * np.where(degree < 3, 1.1, 1.)
                ordinary = raw * np.where(degree < 3, prior['ordinary']['factor'], 1.)
                guard = raw * np.where(degree < 3, prior['guarded']['factor'], 1.)
                values[policy].append({'seed': seed, 'score_sha256': prior['score_sha256'],
                    'checkpoint_sha256': prior['checkpoint_sha256'], 'scores': {'none': raw,
                    'original': z['original'].copy(), 'continuous': z['continuous'].copy(), 'fixed_1_1': fixed,
                    'c_pairwise': ordinary, 'c_guarded': guard}})
    # Every score path is fixed before D labels are materialized.
    csv = project / 'processed_data' / dataset / f'ml_{dataset}.csv'
    if csv.is_symlink():
        raise ValueError('Redirected data')
    data = pd.read_csv(csv, usecols=['ts', 'label'], skiprows=range(1, lo + 1), nrows=hi-lo)
    if (len(data) != hi-lo or set(np.unique(data.label)) != {0, 1}
            or not np.array_equal(data.ts.to_numpy(dtype=np.float64), expected_times)):
        raise ValueError('D labels or timestamps mismatch')
    results, summary = {}, {}
    for policy, entries in values.items():
        results[policy] = []
        prior_items = {x['seed']: x for x in previous['results'][policy]}
        for entry in entries:
            scores = entry.pop('scores'); calculated = {name: metrics(data.label, scores[name]) for name in METHODS}
            prior = prior_items[entry['seed']]['metrics']
            for name, prior_name in (('none', 'none'), ('original', 'original'), ('continuous', 'continuous'),
                                     ('c_pairwise', 'pairwise'), ('c_guarded', 'guarded')):
                if any(abs(calculated[name][key] - prior[prior_name][key]) > 1e-12 for key in METRICS):
                    raise ValueError('Previously reported metrics do not match')
            results[policy].append({**entry, 'metrics': calculated})
        summary[policy] = {}
        for base in ('none', 'original', 'continuous', 'c_pairwise', 'c_guarded'):
            summary[policy][base] = {}
            for metric in METRICS:
                delta = np.array([x['metrics']['fixed_1_1'][metric] - x['metrics'][base][metric] for x in results[policy]])
                summary[policy][base][metric] = {'mean': float(delta.mean()), 'sample_sd': float(delta.std(ddof=1)),
                                                 'per_seed': delta.tolist()}
    _verify_run(project, run_id)
    if (file_sha256(sp) != previous['development_seal_sha256'] or file_sha256(pp) != seal['protocol_sha256']
            or any(file_sha256(dev / f"{policy}_seed{x['seed']}_scores.npz") != x['score_sha256']
                   for policy, entries in results.items() for x in entries)):
        raise RuntimeError('Inputs changed during analysis')
    return {'run_id': run_id, 'development_bounds': [lo, hi], 'development_positives': int(np.sum(data.label == 1)),
            'results': results, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-exploration', action='store_true')
    args = parser.parse_args()
    if not args.confirm_exploration:
        parser.error('Requires --confirm-exploration')
    project = Path(__file__).resolve().parent
    if Path.cwd().resolve() != project:
        raise ValueError('Run from src')
    dest = project / 'research_runs/fixed_degree_up_exploration_20261123'
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('Refusing to overwrite exploration')
    prior_path = project / 'research_runs/c_reference_pairwise_exploration_20261123/report.json'
    if prior_path.is_symlink() or file_sha256(prior_path) != PRIOR:
        raise ValueError('Prior comparison report mismatch')
    previous = json.loads(prior_path.read_text())
    if (set(previous['datasets']) != {'digg', 'bitcoinalpha', 'wikipedia', 'mooc'}
            or any(file_sha256(project / n) != h for n, h in previous['code_sha256'].items())):
        raise ValueError('Prior datasets or code mismatch')
    codehash = file_sha256(Path(__file__).resolve())
    data = {name: analyze(project, name, prior) for name, prior in previous['datasets'].items()}
    if file_sha256(prior_path) != PRIOR or file_sha256(Path(__file__).resolve()) != codehash:
        raise RuntimeError('Inputs changed during analysis')
    report = {'protocol': 'fixed_degree_up_development_v1', 'prior_report_sha256': PRIOR,
        'scope': 'exploratory_previously_observed_D', 'candidate': 'multiply raw score by 1.1 when sampled degree < 3',
        'warning': 'direction considered after exploration; not an independent test; old scores lack edge IDs',
        'label_release_time': 'unknown', 'score_dtype': 'float64 postprocessing; not official torch float32',
        'code_sha256': codehash, 'datasets': data}
    dest.mkdir(exist_ok=False)
    with (dest / 'report.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    for name, d in data.items():
        for policy, s in d['summary'].items():
            print(name, policy, 'fixed 1.1 AUROC vs None/Original/guarded:',
                  [round(s[x]['AUROC']['mean'], 6) for x in ('none', 'original', 'c_guarded')], flush=True)
    print('Saved', dest / 'report.json')


if __name__ == '__main__':
    main()
