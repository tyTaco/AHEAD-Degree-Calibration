"""Exploratory supervised calibration on C; evaluate on previously explored D.

No retraining, model inference, network, or old Test evaluation. This is
retrospective: label release times are unknown, D has already influenced
research, and old score artifacts do not retain event IDs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conservative_pairwise_development import choose_guarded_factor
from degree_development import batch_completion_times
from retrospective_pairwise_development import choose_factor, fixed_scores, evaluate_frozen
from run_label_gate_development import _verify_run
from train_research import file_sha256

RUNS = {'digg': 'digg_shared_dev_5seed_fixed100_20261121',
        'bitcoinalpha': 'bitcoinalpha_shared_dev_5seed_fixed100_20261121',
        'wikipedia': 'wikipedia_shared_dev_5seed_fixed100_20261121',
        'mooc': 'mooc_segmented_dev_5seed_fixed100_20261121'}
POLICIES = ('val_auroc', 'train_loss')
METHODS = ('none', 'original', 'continuous', 'pairwise', 'guarded')
METRICS = ('AUROC', 'AP', 'PR_AUC')


def analyze(project, dataset):
    root, provenance, runs = _verify_run(project, RUNS[dataset])
    if provenance['dataset'] != dataset or sorted(r['seed'] for r in runs) != list(range(5)):
        raise ValueError('Unexpected dataset or seeds')
    dev = root / 'degree_development'
    pp, sp = dev / 'development_started.json', dev / 'development_complete.json'
    if any(x.is_symlink() for x in (dev, pp, sp)):
        raise ValueError('Redirected development artifacts')
    protocol, seal = json.loads(pp.read_text()), json.loads(sp.read_text())
    seal_hash = file_sha256(sp)
    if (file_sha256(pp) != seal['protocol_sha256'] or protocol['dataset'] != dataset
            or protocol['run_id'] != RUNS[dataset] or seal['num_runs'] != 5):
        raise ValueError('Development provenance mismatch')
    c_start, c_stop = protocol['segments']['reference']
    d_start, d_stop = protocol['segments']['development']
    test_start, _ = protocol['segments']['old_test']
    if not 0 <= c_start < c_stop == d_start < d_stop <= test_start:
        raise ValueError('Reference and development must precede old Test')
    csv = project / 'processed_data' / dataset / f'ml_{dataset}.csv'
    if csv.is_symlink() or not csv.is_file():
        raise ValueError('Missing or redirected data')
    # Numeric conversion is limited to C and D, no later label is requested.
    c = pd.read_csv(csv, usecols=['ts', 'label'], skiprows=range(1, c_start + 1), nrows=c_stop-c_start)
    if len(c) != c_stop-c_start or not set(np.unique(c.label)).issubset({0, 1}):
        raise ValueError('Invalid C labels')
    c_times = c.ts.to_numpy(dtype=np.float64)
    choices, scores = {}, {}
    for policy in POLICIES:
        choices[policy], scores[policy] = [], []
        for run in sorted(runs, key=lambda x: x['seed']):
            name = f"{policy}_seed{run['seed']}_scores.npz"; path = dev / name
            if path.is_symlink() or file_sha256(path) != seal['artifact_sha256'][name]:
                raise ValueError('Score artifact mismatch')
            with np.load(path, allow_pickle=False) as z:
                if (not np.array_equal(z['reference_times'], c_times)
                        or not np.array_equal(z['reference_score_available_at'], batch_completion_times(c_times, protocol['batch_size']))
                        or len(z['none']) != d_stop-d_start
                        or np.any(z['reference_score_available_at'] >= z['development_times'][0])):
                    raise ValueError('Reference times, availability or score lengths mismatch')
                ordinary = choose_factor(z['reference_scores'], z['reference_degrees'], c.label.to_numpy())
                guarded = choose_guarded_factor(z['reference_scores'], z['reference_degrees'], c.label.to_numpy())
                methods = fixed_scores(z['none'], z['development_degrees'], ordinary)
                guard = fixed_scores(z['none'], z['development_degrees'],
                                     {'factor': guarded['factor'], 'reason': 'historical_auroc'})['pairwise']
                if (not np.array_equal(methods['original'], z['original'])
                        or not np.array_equal(methods['continuous'], z['continuous'])):
                    raise ValueError('Stored baseline formula mismatch')
                methods['guarded'] = guard
                choices[policy].append({'seed': run['seed'], 'ordinary': ordinary, 'guarded': guarded,
                    'score_sha256': seal['artifact_sha256'][name],
                    'checkpoint_sha256': run['policies'][policy]['checkpoint_sha256']})
                scores[policy].append((z['development_times'].copy(), methods))
    # C-fitted choices and all D scores are fixed before D labels are converted.
    d = pd.read_csv(csv, usecols=['ts', 'label'], skiprows=range(1, d_start + 1), nrows=d_stop-d_start)
    if len(d) != d_stop-d_start or set(np.unique(d.label)) != {0, 1}:
        raise ValueError('D must have both binary classes for these metrics')
    results, summary = {}, {}
    for policy in POLICIES:
        results[policy] = []
        for choice, (times, methods) in zip(choices[policy], scores[policy]):
            if not np.array_equal(times, d.ts.to_numpy(dtype=np.float64)):
                raise ValueError('D event timestamps differ')
            first = {k: methods[k] for k in ('none', 'original', 'continuous', 'pairwise')}
            metrics = evaluate_frozen(first, d.label.to_numpy())
            metrics['guarded'] = evaluate_frozen(dict(first, pairwise=methods['guarded']), d.label.to_numpy())['pairwise']
            results[policy].append({**choice, 'metrics': metrics})
        summary[policy] = {}
        for candidate in ('pairwise', 'guarded'):
            summary[policy][candidate] = {}
            for base in ('none', 'original', 'continuous'):
                summary[policy][candidate][base] = {}
                for metric in METRICS:
                    delta = np.array([x['metrics'][candidate][metric] - x['metrics'][base][metric] for x in results[policy]])
                    summary[policy][candidate][base][metric] = {'mean': float(delta.mean()),
                        'sample_sd': float(delta.std(ddof=1)), 'per_seed': delta.tolist()}
    _verify_run(project, RUNS[dataset])
    if (file_sha256(sp) != seal_hash or file_sha256(pp) != seal['protocol_sha256']
            or any(file_sha256(dev / f"{policy}_seed{x['seed']}_scores.npz") != x['score_sha256']
                   for policy, items in results.items() for x in items)):
        raise ValueError('Inputs changed during analysis')
    return {'run_id': RUNS[dataset], 'development_seal_sha256': seal_hash,
            'reference_bounds': [c_start, c_stop], 'development_bounds': [d_start, d_stop],
            'reference_positives': int(np.sum(c.label == 1)), 'development_positives': int(np.sum(d.label == 1)),
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
    dest = project / 'research_runs/c_reference_pairwise_exploration_20261123'
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('Refusing to overwrite exploration')
    names = ('run_c_reference_pairwise_development.py', 'conservative_pairwise_development.py',
             'calibration_selection_prototype.py', 'retrospective_pairwise_development.py', 'label_gate_development.py')
    code = {name: file_sha256(project / name) for name in names}
    datasets = {name: analyze(project, name) for name in RUNS}
    if any(file_sha256(project / name) != digest for name, digest in code.items()):
        raise ValueError('Code changed during analysis')
    report = {'protocol': 'c_reference_pairwise_development_v1',
        'scope': 'retrospective_supervised_exploration_previously_observed_D', 'label_release_time': 'unknown',
        'warning': 'C uses historical labels; D was already explored; timestamps do not prove event ID alignment',
        'excluded': 'Bitcoin-OTC earlier labels depend on later events', 'parameters': {'factors': [1., .9, 1.1],
        'block_size': 200, 'replicates': 300, 'lower_quantile': .025, 'seed': 20261121},
        'score_dtype': 'float64 postprocessing; not bitwise official torch float32', 'code_sha256': code, 'datasets': datasets}
    dest.mkdir(exist_ok=False)
    with (dest / 'report.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False)
    for name, data in datasets.items():
        for policy, summary in data['summary'].items():
            items = data['results'][policy]
            print(name, policy, 'ordinary/guarded factors', [x['ordinary']['factor'] for x in items],
                  [x['guarded']['factor'] for x in items], 'guarded AUROC vs None/Original',
                  [round(summary['guarded'][base]['AUROC']['mean'], 6) for base in ('none', 'original')], flush=True)
    print('Saved', dest / 'report.json')


if __name__ == '__main__':
    main()
