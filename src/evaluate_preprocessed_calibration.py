"""Evaluate sealed C/D score artifacts without training or model inference.

The legacy timestamp-only option is explicit and cannot establish event identity
within equal timestamps. D is previously explored development, not final Test.
"""
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import prepare_reproducible as prep
from conservative_pairwise_development import choose_guarded_factor
from degree_development import batch_completion_times, chronological_development_split
from retrospective_pairwise_development import choose_factor, fixed_scores
from run_fixed_degree_up_development import metrics
from run_label_gate_development import _verify_run
from train_research import file_sha256

METHODS = ('original', 'none', 'continuous', 'fixed_1_1', 'c_pairwise', 'c_guarded')
POLICIES = ('val_auroc', 'train_loss')
ROOT = Path(__file__).resolve().parent
ALGORITHM_FILES = ('retrospective_pairwise_development.py', 'conservative_pairwise_development.py',
                   'calibration_selection_prototype.py', 'run_fixed_degree_up_development.py',
                   'label_gate_development.py', 'degree_development.py')
VERIFIER_FILES = ('run_label_gate_development.py', 'train_research.py', 'train_shared_research.py',
                  'train_segmented_research.py', 'evaluate_shared_research.py',
                  'verify_segmented_research.py', 'segmented_selection_data.py')


def checked(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or redirected input: {path}')
    return path


def verified_integration(project, run_id, data_mode_required=False, source_override=None):
    """Verify optional Phase 3.1 seals; never accept its DRY-RUN as a run."""
    start, complete = project.parent / 'integration_start.json', project.parent / 'integration_complete.json'
    if not start.exists() and not complete.exists() and not data_mode_required:
        return None
    start, complete = checked(start), checked(complete)
    s, c = json.loads(start.read_text()), json.loads(complete.read_text())
    if (s.get('status') != 'RUNNING' or c.get('status') != 'COMPLETED'
            or s.get('run_id') != run_id or c.get('start_sha256') != file_sha256(start)
            or c.get('selection_sha256') != file_sha256(checked(project / 'research_runs' / run_id / 'selection_complete.json'))):
        raise ValueError('Integration completion seals invalid')
    if Path(s.get('workspace', '')).resolve() != project.resolve():
        raise ValueError('Integration workspace mismatch')
    dataset = s['dataset']
    directory = project / 'processed_data' / dataset
    manifest = checked(directory / 'reproducibility_manifest.json')
    if file_sha256(manifest) != s.get('manifest_sha256'):
        raise ValueError('Preprocessing manifest changed')
    record = json.loads(manifest.read_text())
    if (record.get('dataset') != dataset or record.get('mode') != s['data_mode']
            or record.get('profile_sha256') != s.get('profile_sha256')
            or record.get('source_sha256') != s.get('source_sha256')
            or record.get('artifacts') != s.get('inputs')):
        raise ValueError('Preprocessing manifest provenance mismatch')
    if (s['data_mode'] == 'historical' and record.get('status') != 'PASS-BITWISE') or (
            s['data_mode'] == 'custom' and record.get('status') != 'CUSTOM-VALID'):
        raise ValueError('Preprocessing status mismatch')
    raw_source = Path(source_override) if source_override is not None else Path(s['source'])
    prep.verify_manifest(dataset, s['data_mode'], raw_source, directory)
    for name, sha in s['inputs'].items():
        if Path(name).name != name or file_sha256(checked(directory / name)) != sha:
            raise ValueError('Integrated data hash mismatch')
    for name, sha in s.get('code_sha256', {}).items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or file_sha256(checked(project / path)) != sha:
            raise ValueError('Integrated code hash mismatch')
    return s


def choose_methods(c_scores, c_degrees, c_labels, methods, opts):
    # This function deliberately has no D inputs or D labels.
    pair = choose_factor(c_scores, c_degrees, c_labels) if 'c_pairwise' in methods else None
    guard = choose_guarded_factor(c_scores, c_degrees, c_labels, block_size=opts['block_size'],
                                  replicates=opts['replicates'], seed=opts['bootstrap_seed']) if 'c_guarded' in methods else None
    return pair, guard


def apply_methods(scores, degrees, methods, pair, guard, fixed_multiplier):
    selected = pair if pair is not None else {'factor': 1., 'reason': 'historical_auroc'}
    arrays = fixed_scores(scores, degrees, selected)
    raw, degree = arrays['none'], np.asarray(degrees)
    available = {'none': raw, 'original': arrays['original'], 'continuous': arrays['continuous'],
                 'fixed_1_1': raw * np.where(degree < 3, fixed_multiplier, 1.),
                 'c_pairwise': arrays['pairwise'],
                 'c_guarded': raw * np.where(degree < 3, guard['factor'], 1.) if guard else None}
    return {name: available[name] for name in methods}


def evaluate(project, run_id, dataset, seeds, policies, methods, mode, opts, expected_seal,
             allow_legacy=False, require_integration=False, source_override=None):
    project = Path(project).resolve(strict=True)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', run_id):
        raise ValueError('Invalid run ID')
    integrated = verified_integration(project, run_id, require_integration or project != ROOT.resolve(), source_override)
    root, provenance, runs = _verify_run(project, run_id)
    if provenance.get('dataset') != dataset or (integrated and integrated['dataset'] != dataset):
        raise ValueError('Dataset mismatch')
    if not set(policies).issubset(POLICIES) or not set(methods).issubset(METHODS) or not policies or not methods:
        raise ValueError('Invalid policies or methods')
    algorithm_hashes = {name: file_sha256(checked(ROOT / name)) for name in ALGORITHM_FILES}
    verifier_hashes = {name: file_sha256(checked(ROOT / name)) for name in VERIFIER_FILES}
    for name, sha in {**algorithm_hashes, **verifier_hashes}.items():
        if file_sha256(checked(project / name)) != sha:
            raise ValueError(f'Calibration code differs from workspace: {name}')
    indexed = {item['seed']: item for item in runs}
    if not seeds or not set(seeds).issubset(indexed):
        raise ValueError('Missing seed')
    if integrated and (integrated['configuration'] != provenance['configuration']
                       or integrated['calibration'] != provenance['calibration_mode']):
        raise ValueError('Integration and training configuration mismatch')
    dev = root / 'degree_development'
    if dev.is_symlink() or not dev.is_dir():
        raise ValueError('Missing or redirected score directory')
    pp, sp = checked(dev / 'development_started.json'), checked(dev / 'development_complete.json')
    if not re.fullmatch(r'[0-9a-f]{64}', expected_seal) or file_sha256(sp) != expected_seal:
        raise ValueError('Score seal SHA-256 mismatch')
    protocol, seal = json.loads(pp.read_text()), json.loads(sp.read_text())
    if (protocol.get('run_id') != run_id or protocol.get('dataset') != dataset
            or file_sha256(pp) != seal.get('protocol_sha256')
            or protocol.get('input_provenance_sha256') != file_sha256(checked(root / 'provenance.json'))
            or protocol.get('selection_seal_sha256') != file_sha256(checked(root / 'selection_complete.json'))
            or seal.get('num_runs') != len(runs)):
        raise ValueError('Development provenance mismatch')
    for name, sha in protocol.get('development_code_sha256', {}).items():
        if Path(name).name != name or file_sha256(checked(project / name)) != sha:
            raise ValueError('Scoring code changed')
    csv = checked(project / 'processed_data' / dataset / f'ml_{dataset}.csv')
    metadata = pd.read_csv(csv, usecols=['ts', 'idx'])
    times = metadata.ts.to_numpy(dtype=np.float64)
    ids = metadata.idx.to_numpy(dtype=np.int64)
    segments = chronological_development_split(times)
    if any(protocol['segments'][name] != [x.start, x.stop] for name, x in segments.items()):
        raise ValueError('C/D or old Test boundaries do not match original protocol')
    c, d = segments['reference'], segments['development']
    c_frame = pd.read_csv(csv, usecols=['ts', 'idx', 'label'], skiprows=range(1, c.start + 1), nrows=c.stop-c.start)
    if len(c_frame) != c.stop-c.start or not set(np.unique(c_frame.label)).issubset({0, 1}):
        raise ValueError('Invalid C labels')
    c_times, d_times = times[c], times[d]
    computed = []
    for policy in policies:
        for seed in seeds:
            item = indexed[seed]
            score_name = f'{policy}_seed{seed}_scores.npz'
            metric_name = f'{policy}_seed{seed}_metrics.json'
            score_path, metric_path = checked(dev / score_name), checked(dev / metric_name)
            score_sha, metric_sha = file_sha256(score_path), file_sha256(metric_path)
            if (seal['artifact_sha256'].get(score_name) != score_sha
                    or seal['artifact_sha256'].get(metric_name) != metric_sha):
                raise ValueError('Scores or scoring metadata hash mismatch')
            item_meta = json.loads(metric_path.read_text())
            checkpoint = item['policies'][policy]
            if (item_meta.get('seed') != seed or item_meta.get('policy') != policy
                    or item_meta.get('checkpoint_sha256') != checkpoint['checkpoint_sha256']
                    or item_meta.get('selected_epoch') != checkpoint['epoch']):
                raise ValueError('Score and checkpoint provenance mismatch')
            with np.load(score_path, allow_pickle=False) as z:
                if (not np.array_equal(z['reference_times'], c_times)
                        or not np.array_equal(z['development_times'], d_times)
                        or not np.array_equal(z['reference_score_available_at'], batch_completion_times(c_times, protocol['batch_size']))
                        or not np.array_equal(z['development_score_available_at'], batch_completion_times(d_times, protocol['batch_size']))
                        or z['reference_score_available_at'][-1] >= d_times[0]):
                    raise ValueError('C/D timestamps or batch availability mismatch')
                has_c, has_d = 'reference_event_ids' in z, 'development_event_ids' in z
                if has_c != has_d:
                    raise ValueError('Incomplete event identifiers')
                if has_c:
                    if not np.array_equal(z['reference_event_ids'], ids[c]) or not np.array_equal(z['development_event_ids'], ids[d]):
                        raise ValueError('Event identifiers mismatch')
                    alignment = 'EVENT_IDS_MATCH'
                elif allow_legacy:
                    alignment = 'LEGACY_TIMESTAMPS_ONLY_EVENT_ID_UNVERIFIED'
                else:
                    raise ValueError('Legacy scores lack event IDs; explicit --allow-legacy-event-alignment required')
                pair, guard = choose_methods(z['reference_scores'], z['reference_degrees'], c_frame.label.to_numpy(), methods, opts)
                arrays = apply_methods(z['none'], z['development_degrees'], methods, pair, guard, opts['fixed_multiplier'])
                if ('original' in arrays and not np.array_equal(arrays['original'], z['original'])) or (
                        'continuous' in arrays and not np.array_equal(arrays['continuous'], z['continuous'])):
                    raise ValueError('Stored baseline score mismatch')
            computed.append({'seed': seed, 'policy': policy, 'score_sha256': score_sha,
                             'checkpoint_sha256': checkpoint['checkpoint_sha256'], 'score_metadata_sha256': metric_sha,
                             'selected_epoch': checkpoint['epoch'], 'alignment': alignment,
                             'c_pairwise_selection': pair, 'c_guarded_selection': guard, 'arrays': arrays})
    # D labels are loaded only after all C selections are frozen.
    labels = pd.read_csv(csv, usecols=['ts', 'idx', 'label'], skiprows=range(1, d.start + 1), nrows=d.stop-d.start)
    if (len(labels) != d.stop-d.start or set(np.unique(labels.label)) != {0, 1}
            or not np.array_equal(labels.ts.to_numpy(dtype=np.float64), d_times)
            or not np.array_equal(labels.idx.to_numpy(dtype=np.int64), ids[d])):
        raise ValueError('Invalid D events or labels')
    rows = []
    for entry in computed:
        arrays = entry.pop('arrays')
        entry['metrics'] = {name: metrics(labels.label.to_numpy(), values) for name, values in arrays.items()}
        rows.append(entry)
    _verify_run(project, run_id)
    if file_sha256(sp) != expected_seal or file_sha256(pp) != seal['protocol_sha256']:
        raise ValueError('Seals changed during evaluation')
    for row in rows:
        if file_sha256(dev / f"{row['policy']}_seed{row['seed']}_scores.npz") != row['score_sha256']:
            raise ValueError('Score changed during evaluation')
        if file_sha256(dev / f"{row['policy']}_seed{row['seed']}_metrics.json") != row['score_metadata_sha256']:
            raise ValueError('Score metadata changed during evaluation')
    if any(file_sha256(ROOT / name) != sha or file_sha256(project / name) != sha
           for name, sha in {**algorithm_hashes, **verifier_hashes}.items()):
        raise ValueError('Calibration or verifier code changed during evaluation')
    return {'schema_version': 1, 'status': 'COMPLETED_SCORES_ONLY', 'dataset': dataset, 'run_id': run_id,
            'data_mode': integrated['data_mode'] if integrated else 'HISTORICAL_RUN_NOT_INTEGRATED',
            'experiment_mode': integrated['experiment_mode'] if integrated else 'HISTORICAL_RUN_NOT_INTEGRATED',
            'data_manifest_sha256': integrated['manifest_sha256'] if integrated else None,
            'selection_seal_sha256': protocol['selection_seal_sha256'], 'development_seal_sha256': expected_seal,
            'source_code_sha256': file_sha256(Path(__file__)), 'algorithm_sha256': algorithm_hashes, 'verifier_sha256': verifier_hashes, 'calibration_mode': mode, 'configuration': opts,
            'fixed_method_parameters': {'degree_threshold': 3, 'original_factor': 0.9,
                                        'continuous_alpha': 0.1, 'continuous_tau': 3.0,
                                        'c_pairwise_candidates_and_tie_order': [1.0, 0.9, 1.1],
                                        'guard_candidates_and_tie_order': [0.9, 1.1],
                                        'guard_lower_quantile': 0.025, 'guard_min_valid_fraction': 0.9},
            'methods': list(methods), 'policies': list(policies), 'reference_bounds': [c.start,c.stop],
            'development_bounds': [d.start,d.stop], 'reference_events': len(c_frame),
            'reference_positives': int(c_frame.label.sum()), 'development_events': len(labels),
            'development_positives': int(labels.label.sum()), 'label_scope': 'C supervised; D explored; no old Test evaluation',
            'label_release_time': 'UNKNOWN', 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dataset', required=True, choices=('digg','bitcoinalpha','wikipedia','mooc'))
    parser.add_argument('--seeds', nargs='+', required=True, type=int)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS + ('all',), required=True)
    parser.add_argument('--mode', choices=('historical','custom'), default='historical')
    parser.add_argument('--fixed-multiplier', type=float)
    parser.add_argument('--block-size', type=int)
    parser.add_argument('--replicates', type=int)
    parser.add_argument('--bootstrap-seed', type=int)
    parser.add_argument('--development-seal-sha256', required=True)
    parser.add_argument('--allow-legacy-event-alignment', action='store_true')
    parser.add_argument('--require-integration', action='store_true')
    parser.add_argument('--source', type=Path, help='Override Phase 3.1 raw source location if relocated')
    parser.add_argument('--output', type=Path, required=True, help='new JSON path outside original src')
    a = parser.parse_args(argv)
    try:
        if a.mode == 'historical' and any(x is not None for x in (a.fixed_multiplier,a.block_size,a.replicates,a.bootstrap_seed)):
            raise ValueError('Historical mode forbids parameter overrides')
        opts = {'fixed_multiplier': a.fixed_multiplier if a.fixed_multiplier is not None else 1.1,
                'block_size': a.block_size if a.block_size is not None else 200,
                'replicates': a.replicates if a.replicates is not None else 300,
                'bootstrap_seed': a.bootstrap_seed if a.bootstrap_seed is not None else 20261121}
        if (not math.isfinite(opts['fixed_multiplier']) or opts['fixed_multiplier'] <= 0
                or opts['block_size'] < 1 or opts['replicates'] < 20 or opts['bootstrap_seed'] < 0):
            raise ValueError('Invalid calibration settings')
        methods = METHODS if 'all' in a.methods else tuple(dict.fromkeys(a.methods))
        if len(set(a.seeds)) != len(a.seeds) or len(set(a.policies)) != len(a.policies):
            raise ValueError('Duplicate seeds or policies')
        destination = a.output.absolute()
        if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
            raise ValueError('Output exists, is redirected, or its parent is missing')
        if destination.resolve().is_relative_to(ROOT):
            raise ValueError('Output cannot be written inside original research src')
        result = evaluate(a.project_root, a.run_id, a.dataset, a.seeds, a.policies, methods, a.mode, opts,
                          a.development_seal_sha256, a.allow_legacy_event_alignment, a.require_integration, a.source)
        with destination.open('x', encoding='utf-8') as out:
            json.dump(result, out, indent=2, allow_nan=False); out.write('\n')
        print(json.dumps({'status': result['status'], 'output': str(destination), 'rows': len(result['rows'])}))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'Calibration failed: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
