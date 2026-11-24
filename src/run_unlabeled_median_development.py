"""Retrospective unlabeled degree-median exploration on four known D splits.

This does not download, train, score a model, or request old Test labels. All
four development sets and method design were previously explored. Source label
release times are unknown and CSV/hash operations may read other bytes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from bounded_raw_calibration import BoundedMedianDegreeCalibration, BoundedRawCalibration
from run_label_gate_development import _verify_run
from train_research import file_sha256

RUNS = {'digg': 'digg_shared_dev_5seed_fixed100_20261121',
        'bitcoinalpha': 'bitcoinalpha_shared_dev_5seed_fixed100_20261121',
        'wikipedia': 'wikipedia_shared_dev_5seed_fixed100_20261121',
        'mooc': 'mooc_segmented_dev_5seed_fixed100_20261121'}
POLICIES = ('val_auroc', 'train_loss')
METHODS = ('none', 'original', 'continuous', 'bounded_original', 'bounded_median')
METRICS = ('AUROC', 'AP', 'PR_AUC')


def _metrics(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {'AUROC': float(roc_auc_score(labels, scores)),
            'AP': float(average_precision_score(labels, scores)),
            'PR_AUC': float(auc(recall, precision))}


def analyze(project, dataset):
    root, provenance, runs = _verify_run(project, RUNS[dataset])
    if provenance['dataset'] != dataset or sorted(r['seed'] for r in runs) != list(range(5)):
        raise ValueError('Unexpected dataset or seeds')
    directory = root / 'degree_development'
    protocol_path, seal_path = directory / 'development_started.json', directory / 'development_complete.json'
    if any(p.is_symlink() for p in (directory, protocol_path, seal_path)):
        raise ValueError('Redirected development files')
    protocol, seal = json.loads(protocol_path.read_text()), json.loads(seal_path.read_text())
    seal_hash = file_sha256(seal_path)
    if (file_sha256(protocol_path) != seal['protocol_sha256'] or protocol['dataset'] != dataset
            or protocol['run_id'] != RUNS[dataset] or seal['num_runs'] != 5):
        raise ValueError('Development provenance mismatch')
    lo, hi = protocol['segments']['development']
    if not 0 <= lo < hi <= protocol['segments']['old_test'][0]:
        raise ValueError('Development crosses old Test')
    values, reference_degree, development_degree, development_times = {}, None, None, None
    for policy in POLICIES:
        values[policy] = []
        for run in sorted(runs, key=lambda x: x['seed']):
            name = f"{policy}_seed{run['seed']}_scores.npz"
            path = directory / name
            if path.is_symlink() or file_sha256(path) != seal['artifact_sha256'][name]:
                raise ValueError('Score artifact mismatch')
            with np.load(path, allow_pickle=False) as z:
                raw, degree, ref_degree = z['none'], z['development_degrees'], z['reference_degrees']
                if (len(raw) != hi-lo or len(degree) != hi-lo
                        or not np.array_equal(z['original'], raw.astype(np.float64) * np.where(degree < 3, .9, 1.))):
                    raise ValueError('Score length or original formula mismatch')
                if reference_degree is None:
                    reference_degree, development_degree = ref_degree.copy(), degree.copy()
                    development_times = z['development_times'].copy()
                elif (not np.array_equal(reference_degree, ref_degree)
                      or not np.array_equal(development_degree, degree)
                      or not np.array_equal(development_times, z['development_times'])):
                    raise ValueError('Event axes differ between seed or policy')
                ready = np.float64(protocol['model_ready_time'])
                candidate = BoundedMedianDegreeCalibration(epsilon=.05, min_group=30).fit(
                    z['reference_scores'], ref_degree, z['reference_score_available_at'], model_ready_time=ready)
                result = candidate.transform(raw, degree, z['development_score_available_at'])
                baseline = BoundedRawCalibration(epsilon=.05).fit(z['reference_scores'],
                    z['reference_score_available_at'], model_ready_time=ready).transform(
                    raw, z['original'], z['development_score_available_at'])
                difference = result - raw.astype(np.float64)
                if np.any(np.abs(difference) > candidate._base._limit + 1e-12):
                    raise ValueError('Candidate exceeded shift limit')
                values[policy].append({'seed': run['seed'], 'scores': {'none': raw.copy(),
                    'original': z['original'].copy(), 'continuous': z['continuous'].copy(),
                    'bounded_original': baseline, 'bounded_median': result},
                    'shift': candidate.shift, 'shift_limit': candidate._base._limit,
                    'selection_reason': candidate.reason, 'reference_counts': candidate.reference_counts,
                    'different_from_bounded_original': int(np.count_nonzero(result != baseline)),
                    'score_sha256': seal['artifact_sha256'][name],
                    'checkpoint_sha256': run['policies'][policy]['checkpoint_sha256']})
    # Candidate generation is complete before the D labels are converted.
    csv = project / 'processed_data' / dataset / f'ml_{dataset}.csv'
    if csv.is_symlink():
        raise ValueError('Redirected data')
    data = pd.read_csv(csv, usecols=['ts', 'label'], skiprows=range(1, lo+1), nrows=hi-lo)
    labels = data.label.to_numpy()
    if len(labels) != hi-lo or set(np.unique(labels)) != {0, 1} or not np.array_equal(data.ts.to_numpy(), development_times):
        raise ValueError('Development labels or timestamps mismatch')
    low = development_degree < 3
    groups = {name: {'events': int(np.count_nonzero(mask)), 'positive': int(np.sum(labels[mask] == 1))}
              for name, mask in (('low', low), ('high', ~low))}
    results, summary = {}, {}
    for policy in POLICIES:
        results[policy] = []
        for item in values[policy]:
            scores = item.pop('scores')
            item['metrics'] = {name: _metrics(labels, scores[name]) for name in METHODS}
            results[policy].append(item)
        summary[policy] = {}
        for base in METHODS[:-1]:
            summary[policy][base] = {}
            for metric in METRICS:
                delta = np.array([x['metrics']['bounded_median'][metric] - x['metrics'][base][metric]
                                  for x in results[policy]])
                summary[policy][base][metric] = {'mean': float(delta.mean()),
                    'sample_sd': float(delta.std(ddof=1)), 'per_seed': delta.tolist()}
    _verify_run(project, RUNS[dataset])
    if (file_sha256(seal_path) != seal_hash or file_sha256(protocol_path) != seal['protocol_sha256']
            or any(file_sha256(directory / f"{policy}_seed{x['seed']}_scores.npz") != x['score_sha256']
                   for policy, items in results.items() for x in items)):
        raise ValueError('Inputs changed during analysis')
    return {'run_id': RUNS[dataset], 'development_bounds': [lo, hi],
            'development_seal_sha256': seal_hash, 'groups': groups, 'results': results, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-exploration', action='store_true')
    args = parser.parse_args()
    if not args.confirm_exploration:
        parser.error('Requires --confirm-exploration')
    project = Path(__file__).resolve().parent
    if Path.cwd().resolve() != project:
        raise ValueError('Run from src')
    destination = project / 'research_runs/unlabeled_median_exploration_20261123'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Refusing to overwrite prior exploration')
    names = ('run_unlabeled_median_development.py', 'bounded_raw_calibration.py', 'degree_ecdf_prototype.py')
    code = {name: file_sha256(project / name) for name in names}
    datasets = {name: analyze(project, name) for name in RUNS}
    if any(file_sha256(project / name) != digest for name, digest in code.items()):
        raise ValueError('Code changed during analysis')
    report = {'protocol': 'unlabeled_median_development_v1', 'scope': 'exploratory_previously_observed_D',
        'warning': 'unlabeled C fit; outcome labels on explored D; release time unknown; old score files lack edge IDs',
        'excluded': 'Bitcoin-OTC earlier labels depend on later events',
        'parameters': {'epsilon': .05, 'min_group': 30, 'degree_threshold': 3},
        'score_dtype': 'float64 postprocessing; not official torch float32', 'code_sha256': code, 'datasets': datasets}
    destination.mkdir(exist_ok=False)
    with (destination / 'report.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    for dataset, value in datasets.items():
        for policy, comparisons in value['summary'].items():
            print(dataset, policy, 'AUROC vs None/Original/bounded Original:',
                  [round(comparisons[x]['AUROC']['mean'], 6) for x in ('none', 'original', 'bounded_original')])
    print('Saved', destination / 'report.json')


if __name__ == '__main__':
    main()
