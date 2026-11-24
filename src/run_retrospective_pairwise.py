"""One-shot retrospective Digg development report; never an independent test.

Uses labels from an already explored D interval. Actual label release time is
unknown and Digg labels are synthetic. Never pass old Test labels to choose a
factor. This script performs no model training, GPU inference or Git actions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retrospective_pairwise_development import (choose_factor, chronological_cut,
                                                evaluate_frozen, fixed_scores)
from run_label_gate_development import _verify_run
from train_research import file_sha256


RUN_ID = 'digg_shared_dev_5seed_fixed100_20261121'
PROTOCOL = 'retrospective_digg_pairwise_development_v1'


def run(project, *, confirmed=False):
    if not confirmed:
        raise ValueError('Explicit retrospective development opt-in required')
    project = Path(project).resolve()
    if Path.cwd().resolve() != project:
        raise ValueError('Run from the src directory')
    root, provenance, runs = _verify_run(project, RUN_ID)
    if provenance.get('dataset') != 'digg' or sorted(x['seed'] for x in runs) != list(range(5)):
        raise ValueError('Expected the audited five-seed Digg run')
    dev = root / 'degree_development'
    destination = root / 'retrospective_pairwise_development'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Retrospective report already exists; refusing overwrite')
    protocol_path, seal_path = dev / 'development_started.json', dev / 'development_complete.json'
    if dev.is_symlink() or protocol_path.is_symlink() or seal_path.is_symlink():
        raise ValueError('Redirected development artifacts')
    protocol, seal = json.loads(protocol_path.read_text()), json.loads(seal_path.read_text())
    seal_hash = file_sha256(seal_path)
    if (file_sha256(protocol_path) != seal.get('protocol_sha256')
            or protocol.get('run_id') != RUN_ID or protocol.get('dataset') != 'digg'
            or protocol.get('input_provenance_sha256') != file_sha256(root / 'provenance.json')
            or protocol.get('selection_seal_sha256') != file_sha256(root / 'selection_complete.json')
            or seal.get('num_runs') != 5 or protocol.get('batch_size') != 200):
        raise ValueError('Development provenance does not match the audited run')
    lo, hi = protocol['segments']['development']
    test_lo, _ = protocol['segments']['old_test']
    if not isinstance(lo, int) or not isinstance(hi, int) or not 0 <= lo < hi <= test_lo:
        raise ValueError('Development segment must precede the old Test segment')
    csv = project / 'processed_data/digg/ml_digg.csv'
    if csv.is_symlink() or not csv.is_file():
        raise ValueError('Missing or redirected data')
    data = pd.read_csv(csv, usecols=['label', 'ts'], skiprows=range(1, lo + 1), nrows=hi-lo)
    labels, data_times = data['label'].to_numpy(), data['ts'].to_numpy(dtype=np.float64)
    if len(labels) != hi-lo or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Invalid development labels')
    artifacts = seal.get('artifact_sha256', {})
    code_paths = [project / name for name in ('run_retrospective_pairwise.py',
        'retrospective_pairwise_development.py', 'label_gate_development.py')]
    code_hashes = {p.name: file_sha256(p) for p in code_paths}
    results, cut_expected, degree_expected = {}, None, None
    for policy in ('val_auroc', 'train_loss'):
        results[policy] = []
        for item in sorted(runs, key=lambda x: x['seed']):
            name = f"{policy}_seed{item['seed']}_scores.npz"
            path = dev / name
            if path.is_symlink() or name not in artifacts or file_sha256(path) != artifacts[name]:
                raise ValueError('Development score artifact changed')
            with np.load(path, allow_pickle=False) as z:
                raw, degree = z['none'], z['development_degrees']
                if (not np.array_equal(z['development_times'], data_times)
                        or (degree_expected is not None and not np.array_equal(degree, degree_expected))):
                    raise ValueError('Development event timestamps or degrees do not align')
                if degree_expected is None:
                    degree_expected = degree.copy()
                cut = chronological_cut(z['development_times'], z['development_score_available_at'], batch_size=200)
                if len(raw) != len(labels) or (cut_expected is not None and cut != cut_expected):
                    raise ValueError('Score/label alignment or split mismatch')
                cut_expected = cut
                selection = choose_factor(raw[:cut], degree[:cut], labels[:cut])
                scores = fixed_scores(raw[cut:], degree[cut:], selection)
                if (not np.array_equal(scores['original'], z['original'][cut:])
                        or not np.array_equal(scores['continuous'], z['continuous'][cut:])):
                    raise ValueError('Stored comparison paths do not match reconstructed scores')
                metrics = evaluate_frozen(scores, labels[cut:])
            results[policy].append({'seed': item['seed'], 'selection': selection, 'metrics': metrics,
                'checkpoint_sha256': item['policies'][policy]['checkpoint_sha256'], 'score_sha256': artifacts[name]})
    summary = {}
    for policy, items in results.items():
        summary[policy] = {'selected_factors': [x['selection']['factor'] for x in items], 'paired_differences': {}}
        for baseline in ('none', 'original', 'continuous'):
            summary[policy]['paired_differences'][baseline] = {}
            for metric in ('AUROC', 'AP', 'PR_AUC'):
                delta = np.array([x['metrics']['pairwise'][metric] - x['metrics'][baseline][metric] for x in items])
                summary[policy]['paired_differences'][baseline][metric] = {
                    'mean': float(delta.mean()), 'sample_sd': float(delta.std(ddof=1))}
    _verify_run(project, RUN_ID)
    if (file_sha256(seal_path) != seal_hash or file_sha256(protocol_path) != seal['protocol_sha256']
            or any(file_sha256(dev / f"{policy}_seed{item['seed']}_scores.npz") != item['score_sha256']
                   for policy, items in results.items() for item in items)
            or any(file_sha256(p) != code_hashes[p.name] for p in code_paths)):
        raise ValueError('Inputs changed during analysis')
    report = {'protocol': PROTOCOL, 'run_id': RUN_ID, 'dataset': 'digg',
        'scope': 'retrospective_supervised_development_not_independent_test',
        'label_release_time': 'unknown', 'labels': 'synthetic; earlier labels are used for factor selection',
        'selection_cut_index': cut_expected, 'selection_count': cut_expected,
        'evaluation_count': len(labels) - cut_expected, 'batch_size': 200,
        'factors': [1., .9, 1.1], 'score_dtype': 'float64 postprocessing; differs from author torch float32',
        'alignment_limit': 'timestamps and degrees checked; old score artifacts do not retain edge IDs',
        'development_seal_sha256': seal_hash, 'code_sha256': code_hashes,
        'results': results, 'summary': summary}
    destination.mkdir(exist_ok=False)
    with (destination / 'report.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-retrospective-development', action='store_true')
    args = parser.parse_args()
    report = run(Path(__file__).resolve().parent, confirmed=args.confirm_retrospective_development)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
