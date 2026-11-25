"""Isolated, manifest-verified bridge to the existing research trainers.

No experiment is started unless --execute and an explicit confirmation are both
provided. Historical training provenance remains independent and untouched.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import prepare_reproducible as prep

ROOT = Path(__file__).resolve().parent
BASELINES = {
    'wikipedia': ('shared', 'wikipedia_shared_dev_5seed_fixed100_20261121'),
    'bitcoinalpha': ('shared', 'bitcoinalpha_shared_dev_5seed_fixed100_20261121'),
    'mooc': ('segmented', 'mooc_segmented_dev_5seed_fixed100_20261121'),
}


def digest(path):
    return prep.sha256(path)


def check_legacy(directory, dataset):
    """Retain segmented trainer's conversion_manifest contract, if present."""
    p = directory / 'conversion_manifest.json'
    if not p.exists() and not p.is_symlink():
        return 'ABSENT'
    if p.is_symlink():
        raise ValueError('Legacy conversion manifest must not be a symlink')
    record = json.loads(p.read_text(encoding='utf-8'))
    names = {f'ml_{dataset}.csv', f'ml_{dataset}.npy', f'ml_{dataset}_node.npy'}
    if (record.get('schema_version') != 1 or set(record.get('artifacts', {})) != names
            or record.get('converter_sha256') != digest(ROOT / 'prepare_jodie_research.py')):
        raise ValueError('Legacy conversion manifest or converter mismatch')
    for name in names:
        p = directory / name
        if p.is_symlink() or digest(p) != record['artifacts'][name]:
            raise ValueError(f'Legacy converted data mismatch: {name}')
    return 'VERIFIED'


def validate(dataset, mode, source, data, manifest_hash):
    if dataset == 'digg':
        raise ValueError('BLOCKED: Digg preprocessing is not established')
    if not re.fullmatch(r'[0-9a-f]{64}', manifest_hash):
        raise ValueError('Expected manifest SHA-256 must be 64 lowercase hex characters')
    data = Path(data)
    manifest = data / 'reproducibility_manifest.json'
    if data.is_symlink() or manifest.is_symlink() or digest(manifest) != manifest_hash:
        raise ValueError('Manifest SHA-256 mismatch or symlink not allowed')
    record = prep.verify_manifest(dataset, mode, Path(source), data)
    if digest(manifest) != manifest_hash:
        raise ValueError('Manifest changed during validation')
    legacy = check_legacy(data, dataset)
    return record, legacy


def configuration(dataset, device, overrides):
    strategy, baseline_id = BASELINES[dataset]
    p = ROOT / 'research_runs' / baseline_id / 'provenance.json'
    provenance = json.loads(p.read_text(encoding='utf-8'))
    if provenance['dataset'] != dataset:
        raise ValueError('Baseline provenance dataset mismatch')
    config = dict(provenance['configuration'])
    if any(k not in ('num_epochs', 'num_runs', 'batch_size', 'learning_rate',
                     'num_neighbors', 'time_gap', 'optimizer', 'weight_decay', 'dropout',
                     'val_ratio', 'test_ratio') for k in overrides):
        raise ValueError('Unsupported training override')
    config.update(overrides)
    config['device'] = device
    if (min(config[k] for k in ('num_epochs', 'num_runs', 'batch_size', 'num_neighbors')) < 1
            or not math.isfinite(config['learning_rate']) or config['learning_rate'] <= 0
            or config['time_gap'] <= 0 or not math.isfinite(config['weight_decay'])
            or config['weight_decay'] < 0 or not math.isfinite(config['dropout'])
            or not 0 <= config['dropout'] < 1 or not math.isfinite(config['val_ratio'])
            or not math.isfinite(config['test_ratio']) or not 0 < config['val_ratio'] < 1
            or not 0 < config['test_ratio'] < 1 or config['val_ratio'] + config['test_ratio'] >= 1
            or config['optimizer'] not in ('Adam', 'SGD', 'RMSprop')):
        raise ValueError('Training values must be positive')
    return strategy, p, config


def code_files():
    # Copy Python sources as independent regular files, never hardlink or symlink.
    files = list(ROOT.glob('*.py'))
    for name in ('models', 'utils'):
        files += list((ROOT / name).rglob('*.py'))
    if any(p.is_symlink() for p in files):
        raise ValueError('Symlinked research source is not supported')
    return sorted(files)


def copy_exclusive(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as inp, Path(destination).open('xb') as out:
        shutil.copyfileobj(inp, out, length=1024 * 1024)


def make_workspace(base, run_id, dataset, source, data, manifest_hash, mode):
    base = Path(base)
    if not base.is_dir() or base.is_symlink():
        raise ValueError('Workspace parent must be an existing non-symlink directory')
    base = base.resolve(strict=True)
    if base == ROOT or base.is_relative_to(ROOT):
        raise ValueError('Workspace must be outside the original src research directory')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', run_id) or run_id in ('.', '..'):
        raise ValueError('Invalid run ID')
    run = base / run_id
    run.mkdir(exist_ok=False)
    project = run / 'project'
    project.mkdir()
    code_hashes = {}
    for item in code_files():
        relative = item.relative_to(ROOT)
        target = project / relative
        before = digest(item)
        copy_exclusive(item, target)
        if digest(target) != before or digest(item) != before:
            raise ValueError(f'Source changed while copying: {relative}')
        code_hashes[str(relative)] = before
    target_data = project / 'processed_data' / dataset
    target_data.mkdir(parents=True)
    names = [f'ml_{dataset}.csv', f'ml_{dataset}.npy', f'ml_{dataset}_node.npy',
             'reproducibility_manifest.json']
    if (Path(data) / 'conversion_manifest.json').exists():
        names.append('conversion_manifest.json')
    for name in names:
        origin = Path(data) / name
        if origin.is_symlink():
            raise ValueError(f'Symlinked data not accepted: {name}')
        before = digest(origin)
        copy_exclusive(origin, target_data / name)
        if digest(target_data / name) != before or digest(origin) != before:
            raise ValueError(f'Data changed during copy: {name}')
        (target_data / name).chmod(0o444)
    validate(dataset, mode, source, target_data, manifest_hash)
    return run, project, target_data, code_hashes


def write_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def invoke(project, strategy, config, run_id, calibration):
    # Original trainers run as copied modules with their historical code hashes.
    script = ("import argparse,json,os,sys; "
              "a=argparse.Namespace(**json.loads(sys.argv[1])); "
              "s=sys.argv[2]; rid=sys.argv[3]; mode=sys.argv[4]; "
              "os.environ['AHEAD_RESEARCH_RUN_ID']=rid; "
              "os.environ['AHEAD_DEGREE_CALIB_MODE']=mode; "
              "exec(\"if s == 'segmented':\\n import train_segmented_research as t\\n t.run(os.getcwd(), a, rid, mode, True)\\nelse:\\n import train_shared_research as t\\n t.get_node_classification_args = lambda: a\\n os.environ['AHEAD_SHARED_TRAINING_CONFIRM']='1'\\n t.main()\")")
    cmd = [sys.executable, '-B', '-c', script, json.dumps(config, allow_nan=False), strategy, run_id, calibration]
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    if config['device'] == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run(cmd, cwd=project, env=env, check=False)
    if result.returncode:
        raise RuntimeError(f'Trainer failed with exit code {result.returncode}')
    return cmd


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=['wikipedia', 'mooc', 'bitcoinalpha', 'digg'])
    p.add_argument('--data-mode', required=True, choices=['historical', 'custom'])
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--data-dir', required=True, type=Path)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--experiment-mode', choices=['historical', 'custom'], default='historical')
    p.add_argument('--device', default='cpu', help='cpu or cuda:N; no device is initialized by dry-run')
    p.add_argument('--calibration', choices=['original', 'none', 'continuous'], default='original')
    p.add_argument('--run-id', required=True)
    p.add_argument('--workspace-parent', type=Path)
    for key, typ in [('num-epochs', int), ('num-runs', int), ('batch-size', int),
                     ('learning-rate', float), ('num-neighbors', int), ('time-gap', int),
                     ('weight-decay', float), ('dropout', float), ('val-ratio', float), ('test-ratio', float)]:
        p.add_argument('--' + key, type=typ)
    p.add_argument('--optimizer', choices=['Adam', 'SGD', 'RMSprop'])
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-only', action='store_true')
    group.add_argument('--dry-run', action='store_true')
    group.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    try:
        if not re.fullmatch(r'cpu|cuda:[0-9]+', args.device):
            raise ValueError('Device must be cpu or cuda:N')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
            raise ValueError('Invalid run ID')
        keys = ('num_epochs', 'num_runs', 'batch_size', 'learning_rate', 'num_neighbors', 'time_gap',
                'optimizer', 'weight_decay', 'dropout', 'val_ratio', 'test_ratio')
        overrides = {k: getattr(args, k) for k in keys if getattr(args, k) is not None}
        if args.experiment_mode == 'historical' and (overrides or args.calibration != 'original'):
            raise ValueError('Historical experiment does not allow parameter or calibration overrides')
        if args.execute and os.environ.get('AHEAD_INTEGRATION_TRAINING_CONFIRM') != '1':
            raise ValueError('Explicit training authorization and AHEAD_INTEGRATION_TRAINING_CONFIRM=1 required')
        record, legacy = validate(args.dataset, args.data_mode, args.source, args.data_dir, args.manifest_sha256)
        strategy, baseline, config = configuration(args.dataset, args.device, overrides)
        info = {'schema_version': 1, 'status': 'VERIFIED', 'run_id': args.run_id,
                'dataset': args.dataset, 'data_mode': args.data_mode, 'experiment_mode': args.experiment_mode,
                'data_status': record['status'], 'historical_method_note': record['provenance_note'],
                'source': str(args.source.resolve()), 'source_sha256': record['source_sha256'],
                'manifest_sha256': args.manifest_sha256, 'profile_sha256': record['profile_sha256'],
                'inputs': record['artifacts'], 'legacy_conversion_manifest': legacy,
                'baseline_provenance': str(baseline), 'baseline_provenance_sha256': digest(baseline),
                'strategy': strategy, 'selection_policies': ['val_auroc', 'train_loss'],
                'configuration': config, 'calibration': args.calibration,
                'seeds': list(range(config['num_runs'])), 'device': args.device,
                'historical_device_changed': args.device != json.loads(baseline.read_text())['configuration']['device'],
                'split': 'trainer val_ratio/test_ratio; D is explored development, not independent Test',
                'rng': 'existing set_research_seed: Python, NumPy and PyTorch; CUDA branch also sets cudnn deterministic and benchmark false; CPU branch does not',
                'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                                'pandas': pd.__version__, 'torch': str(torch.__version__),
                                'pytorch_cuda_build': torch.version.cuda,
                                'scipy': importlib.metadata.version('scipy'),
                                'scikit_learn': importlib.metadata.version('scikit-learn'),
                                'gpu_hardware_and_driver': 'NOT-PROBED'},
                'command': [sys.executable, str(Path(__file__).resolve())] + (argv if argv is not None else sys.argv[1:])}
        if args.verify_only:
            print(json.dumps(info, indent=2, allow_nan=False)); return 0
        if args.workspace_parent is None:
            raise ValueError('--workspace-parent is required for dry-run or execute')
        run, project, copied, codes = make_workspace(args.workspace_parent, args.run_id, args.dataset,
                                                      args.source, args.data_dir, args.manifest_sha256, args.data_mode)
        info['code_sha256'] = codes
        info['workspace'] = str(project)
        info['status'] = 'DRY-RUN' if args.dry_run else 'RUNNING'
        write_new(run / 'integration_start.json', info)
        if args.dry_run:
            print(json.dumps({'status': 'DRY-RUN', 'workspace': str(project), 'strategy': strategy,
                              'trainer_invoked': False}, indent=2)); return 0
        try:
            cmd = invoke(project, strategy, config, args.run_id, args.calibration)
            validate(args.dataset, args.data_mode, args.source, copied, args.manifest_sha256)
            for relative, value in codes.items():
                if digest(project / relative) != value:
                    raise RuntimeError(f'Copied source changed: {relative}')
            seal = project / 'research_runs' / args.run_id / 'selection_complete.json'
            if not seal.is_file():
                raise RuntimeError('Trainer completed without selection seal')
            write_new(run / 'integration_complete.json', {'status': 'COMPLETED', 'selection_sha256': digest(seal),
                                                          'start_sha256': digest(run / 'integration_start.json'), 'command': cmd})
        except Exception as exc:
            write_new(run / 'integration_failed.json', {'status': 'FAILED', 'error': str(exc)})
            raise
        print(json.dumps({'status': 'COMPLETED', 'workspace': str(project)}, indent=2)); return 0
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f'Integration failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
