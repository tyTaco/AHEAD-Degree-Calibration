"""Versioned preprocessing entry; preserves historical converters and outputs.

Historical means byte-level reproduction, not proof of the original algorithm.
The Digg algorithm remains blocked. Outputs must be created in a new directory.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_jodie_research import prepare as prepare_jodie

ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / 'preprocessing_profiles.json'
PROTECTED = tuple(ROOT / name for name in
                  ('processed_data', 'raw_data', 'saved_models', 'saved_scores', 'saved_results', 'research_runs'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_profiles():
    with PROFILES.open(encoding='utf-8') as stream:
        data = json.load(stream)
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported preprocessing profile version')
    return data['datasets']


def copy_new(source, dest):
    with Path(source).open('rb') as src, Path(dest).open('xb') as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def alpha_candidate(source, directory):
    """Reconstructed rule verified against one observed Alpha dataset, not historical provenance."""
    raw = pd.read_csv(source, header=None, names=['u', 'i', 'rating', 'ts'])
    if raw.empty or raw.isna().any().any():
        raise ValueError('Invalid or empty Bitcoin-Alpha raw CSV')
    for key in ('u', 'i', 'rating', 'ts'):
        raw[key] = pd.to_numeric(raw[key], errors='raise')
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Non-finite Bitcoin-Alpha input')
    raw = raw.sort_values('ts', kind='stable').reset_index(drop=True)
    ids = pd.unique(pd.concat([raw.u, raw.i], ignore_index=True))
    mapping = {value: idx + 1 for idx, value in enumerate(ids)}
    graph = pd.DataFrame({'u': raw.u.map(mapping), 'i': raw.i.map(mapping),
                          'ts': raw.ts.astype(float), 'label': (raw.rating < 0).astype(int),
                          'idx': np.arange(1, len(raw) + 1)})
    directory.mkdir()
    with (directory / 'ml_bitcoinalpha.csv').open('x', encoding='utf-8', newline='') as out:
        graph.to_csv(out, index=False)
    for name, shape in [('ml_bitcoinalpha.npy', (len(graph) + 1, 172)),
                        ('ml_bitcoinalpha_node.npy', (len(ids) + 1, 172))]:
        with (directory / name).open('xb') as out:
            np.save(out, np.zeros(shape, dtype=np.float32), allow_pickle=False)
    return {'reconstructed_candidate': True, 'original_historical_algorithm': 'UNKNOWN'}


def inspect_data(directory, dataset):
    graph = directory / f'ml_{dataset}.csv'
    nodes, count, positives, previous = set(), 0, 0, None
    with graph.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['u', 'i', 'ts', 'label', 'idx']:
            raise ValueError('Unexpected CSV schema')
        for row in reader:
            u, i, t, y, idx = int(row['u']), int(row['i']), float(row['ts']), int(row['label']), int(row['idx'])
            count += 1
            if (not np.isfinite(t) or u <= 0 or i <= 0 or idx != count or y not in (0, 1)
                    or (previous is not None and t < previous)):
                raise ValueError('Invalid event, label, index, or timestamp order')
            previous = t
            positives += y
            nodes.update((u, i))
    arrays = {}
    for key, suffix in (('edge', '.npy'), ('node', '_node.npy')):
        arr = np.load(directory / f'ml_{dataset}{suffix}', mmap_mode='r', allow_pickle=False)
        if arr.ndim != 2 or arr.dtype != np.float32 or arr.shape[1] > 172:
            raise ValueError(f'Invalid {key} feature shape or dtype')
        if (key == 'edge' and arr.shape[0] != count + 1) or (key == 'node' and max(nodes, default=0) >= arr.shape[0]):
            raise ValueError(f'Invalid {key} feature index range')
        if any(not np.isfinite(arr[i:i + 2048]).all() for i in range(0, len(arr), 2048)):
            raise ValueError(f'Non-finite {key} features')
        arrays[key] = {'shape': list(arr.shape), 'dtype': str(arr.dtype),
                       'order': 'F' if arr.flags.f_contiguous and not arr.flags.c_contiguous else 'C'}
    return {'csv_schema': ['u', 'i', 'ts', 'label', 'idx'], 'events': count,
            'distinct_endpoints': len(nodes), 'positives': positives, 'features': arrays}


def convert(dataset, mode, source, destination, edge_order=None, node_order=None):
    profiles = load_profiles()
    if dataset not in profiles:
        raise ValueError('Unknown dataset')
    profile = profiles[dataset]
    if profile['status'] == 'BLOCKED':
        raise ValueError(f'BLOCKED: {dataset} historical preprocessing is unknown')
    if mode not in ('historical', 'custom'):
        raise ValueError('Invalid mode')
    if mode == 'historical' and (edge_order is not None or node_order is not None):
        raise ValueError('Historical mode does not accept order overrides')
    edge_order = edge_order or profile['edge_order']
    node_order = node_order or profile['node_order']
    if edge_order not in ('C', 'F') or node_order not in ('C', 'F'):
        raise ValueError('Order must be C or F')
    source = Path(source).resolve(strict=True)
    dest = Path(destination).absolute()
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('Destination already exists; refusing to overwrite')
    resolved = dest.resolve()
    if any(resolved == p.resolve() or resolved.is_relative_to(p.resolve()) for p in PROTECTED):
        raise ValueError('Destination is inside protected research assets')
    if not dest.parent.is_dir():
        raise FileNotFoundError('Destination parent directory must already exist')
    before = sha256(source)
    if mode == 'historical' and before != profile['source_sha256']:
        raise ValueError('Historical source SHA-256 mismatch')
    converter = ROOT / 'prepare_jodie_research.py'
    if mode == 'historical' and dataset in ('wikipedia', 'mooc') and sha256(converter) != profile['converter_sha256']:
        raise ValueError('Historical converter SHA-256 mismatch')
    dest.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=dest) as tmp:
        stage = Path(tmp) / 'stage'
        if dataset == 'bitcoinalpha':
            source_manifest = alpha_candidate(source, stage)
        else:
            source_manifest = prepare_jodie(source, stage, dataset)
        copy_new(stage / f'ml_{dataset}.csv', dest / f'ml_{dataset}.csv')
        for suffix, order in (('.npy', edge_order), ('_node.npy', node_order)):
            name = f'ml_{dataset}{suffix}'
            arr = np.load(stage / name, mmap_mode='r', allow_pickle=False)
            data = np.ascontiguousarray(arr) if order == 'C' else np.asfortranarray(arr)
            with (dest / name).open('xb') as stream:
                np.save(stream, data, allow_pickle=False)
        observed = inspect_data(dest, dataset)
        files = {name: sha256(dest / name) for name in profile['expected_sha256']}
        if sha256(source) != before:
            raise RuntimeError('Source changed during conversion')
        if mode == 'historical':
            expected_stats = (profile['events'], profile['nodes'], profile['positives'])
            stats = (observed['events'], observed['distinct_endpoints'], observed['positives'])
            expected_shapes = ([profile['events'] + 1, profile['edge_dim']],
                               [observed['features']['node']['shape'][0], profile['node_dim']])
            ok = (files == profile['expected_sha256'] and stats == expected_stats
                  and observed['features']['edge']['shape'] == expected_shapes[0]
                  and observed['features']['node']['shape'] == expected_shapes[1]
                  and observed['features']['edge']['order'] == edge_order
                  and observed['features']['node']['order'] == node_order)
            status = 'PASS-BITWISE' if ok else 'MISMATCH'
        else:
            status = 'CUSTOM-VALID'
        manifest = {'schema_version': 1, 'mode': mode, 'dataset': dataset, 'status': status,
                    'historical_reconstruction_status': status if mode == 'historical' else 'NOT-TESTED',
                    'provenance_note': profile['status'], 'source_name': source.name,
                    'source_sha256': before, 'profile_sha256': sha256(PROFILES),
                    'entry_sha256': sha256(Path(__file__)),
                    'underlying_converter_sha256': sha256(converter) if dataset != 'bitcoinalpha' else None,
                    'environment': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__},
                    'parameters': {'edge_order': edge_order, 'node_order': node_order,
                                   'mapping': profile['mapping'], 'sort': profile['sort'],
                                   'ties': profile['ties'], 'label': profile['label'],
                                   'injection': profile['injection'], 'seed': profile['seed'],
                                   'rng': profile['rng'], 'feature_dtype': profile['feature_dtype']},
                    'observed': observed, 'artifacts': files,
                    'expected_historical_sha256': profile['expected_sha256'],
                    'underlying_conversion': source_manifest}
        with (dest / 'reproducibility_manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
        if status == 'MISMATCH':
            raise ValueError('Historical output mismatch; inspect reproducibility_manifest.json')
        return manifest


def verify_manifest(dataset, mode, source, destination):
    """Revalidate saved manifest, source, profiles, implementation, outputs and schema."""
    profile = load_profiles()[dataset]
    if profile['status'] == 'BLOCKED':
        raise ValueError(f'BLOCKED: {dataset} historical preprocessing is unknown')
    dest = Path(destination)
    manifest_path = dest / 'reproducibility_manifest.json'
    if dest.is_symlink() or manifest_path.is_symlink():
        raise ValueError('Symlinked output or manifest is not accepted')
    with manifest_path.open(encoding='utf-8') as stream:
        record = json.load(stream)
    if record.get('schema_version') != 1 or record.get('dataset') != dataset or record.get('mode') != mode:
        raise ValueError('Manifest identity mismatch')
    if record.get('profile_sha256') != sha256(PROFILES) or record.get('entry_sha256') != sha256(Path(__file__)):
        raise ValueError('Profile or entry SHA-256 mismatch')
    if sha256(source) != record.get('source_sha256'):
        raise ValueError('Manifest source SHA-256 mismatch')
    if dataset in ('mooc', 'wikipedia') and record.get('underlying_converter_sha256') != sha256(ROOT / 'prepare_jodie_research.py'):
        raise ValueError('Underlying converter SHA-256 mismatch')
    names = set(profile['expected_sha256'])
    if set(record.get('artifacts', {})) != names:
        raise ValueError('Manifest artifact list mismatch')
    for name in names:
        path = dest / name
        if path.is_symlink() or sha256(path) != record['artifacts'][name]:
            raise ValueError(f'Artifact SHA-256 mismatch: {name}')
    observed = inspect_data(dest, dataset)
    if observed != record.get('observed'):
        raise ValueError('Manifest schema or data statistics mismatch')
    if any(observed['features'][k]['order'] != record['parameters'][k + '_order'] for k in ('edge', 'node')):
        raise ValueError('Manifest array order mismatch')
    if mode == 'historical':
        if (record.get('status') != 'PASS-BITWISE' or record.get('historical_reconstruction_status') != 'PASS-BITWISE'
                or record['source_sha256'] != profile['source_sha256']
                or record['artifacts'] != profile['expected_sha256']
                or record['parameters']['edge_order'] != profile['edge_order']
                or record['parameters']['node_order'] != profile['node_order']):
            raise ValueError('Historical output verification failed')
    elif record.get('status') != 'CUSTOM-VALID' or record.get('historical_reconstruction_status') != 'NOT-TESTED':
        raise ValueError('Custom manifest status mismatch')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('digg', 'bitcoinalpha', 'wikipedia', 'mooc'))
    parser.add_argument('--mode', required=True, choices=('historical', 'custom'))
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--edge-order', choices=('C', 'F'))
    parser.add_argument('--node-order', choices=('C', 'F'))
    parser.add_argument('--verify-only', action='store_true', help='Verify a completed output without writing')
    args = parser.parse_args()
    try:
        if args.verify_only:
            if args.edge_order or args.node_order:
                raise ValueError('Order overrides are not accepted with --verify-only')
            result = verify_manifest(args.dataset, args.mode, args.source, args.destination)
        else:
            result = convert(args.dataset, args.mode, args.source, args.destination, args.edge_order, args.node_order)
        print(json.dumps({'dataset': args.dataset, 'mode': args.mode, 'status': result['status'],
                          'historical_reconstruction_status': result['historical_reconstruction_status']}, indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'Preprocessing failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
