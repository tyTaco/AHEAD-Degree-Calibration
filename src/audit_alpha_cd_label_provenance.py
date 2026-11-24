"""Read-only audit of local Bitcoin-Alpha C/D labels and model features.

Raw endpoint/time metadata is read for the full input to reproduce sorting;
only C/D rating and label values are converted to numeric values. CSV readers,
hashing and the OS still read/buffer other bytes and parse raw fields as text.
This establishes no fraud truth, label release time, or independent holdout.
"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_label_gate_development import _verify_run
from train_research import file_sha256

RUN = 'bitcoinalpha_shared_dev_5seed_fixed100_20261121'


def audit(project):
    project = Path(project).resolve()
    root, provenance, runs = _verify_run(project, RUN)
    if provenance['dataset'] != 'bitcoinalpha' or sorted(x['seed'] for x in runs) != list(range(5)):
        raise ValueError('Unexpected dataset or seeds')
    dev = root / 'degree_development'
    pp, sp = dev / 'development_started.json', dev / 'development_complete.json'
    if any(x.is_symlink() for x in (dev, pp, sp)):
        raise ValueError('Redirected development artifacts')
    protocol, seal = json.loads(pp.read_text()), json.loads(sp.read_text())
    seal_hash = file_sha256(sp)
    if (file_sha256(pp) != seal['protocol_sha256'] or protocol['dataset'] != 'bitcoinalpha'
            or protocol['run_id'] != RUN or seal['num_runs'] != 5):
        raise ValueError('Development provenance mismatch')
    cs, ce = protocol['segments']['reference']; ds, de = protocol['segments']['development']
    test_start, _ = protocol['segments']['old_test']
    if not 0 <= cs < ce == ds < de <= test_start:
        raise ValueError('C/D overlaps old Test')
    source = project / 'raw_data/soc-sign-bitcoinalpha.csv.gz'
    base = project / 'processed_data/bitcoinalpha'
    processed, edge_path, node_path = (base / x for x in ('ml_bitcoinalpha.csv', 'ml_bitcoinalpha.npy', 'ml_bitcoinalpha_node.npy'))
    if any(x.is_symlink() or not x.is_file() for x in (source, processed, edge_path, node_path)):
        raise ValueError('Missing or redirected input')
    source_hash, processed_hash = file_sha256(source), file_sha256(processed)
    raw = pd.read_csv(source, header=None, usecols=[0, 1, 3], names=['src', 'dst', 'ts'])
    raw['raw_line'] = np.arange(len(raw))
    raw = raw.sort_values('ts', kind='mergesort').reset_index(drop=True)
    meta = pd.read_csv(processed, usecols=['u', 'i', 'ts', 'idx'])
    if (len(raw) != len(meta) or not np.array_equal(raw.ts.to_numpy(dtype=np.float64), meta.ts.to_numpy(dtype=np.float64))
            or not np.array_equal(meta.idx.to_numpy(), np.arange(1, len(meta) + 1))):
        raise ValueError('Source and processed event metadata do not align')
    forward, reverse = {}, {}
    for original, local in zip(np.r_[raw.src.to_numpy(), raw.dst.to_numpy()], np.r_[meta.u.to_numpy(), meta.i.to_numpy()]):
        if (original in forward and forward[original] != local) or (local in reverse and reverse[local] != original):
            raise ValueError('Node mapping is not one-to-one')
        forward[original], reverse[local] = local, original
    duplicates = raw.groupby(['src', 'dst', 'ts'], sort=False)['raw_line'].transform('size')
    if (duplicates.iloc[cs:de] > 1).any():
        raise ValueError('Duplicate source/target/time keys in C/D make rating alignment ambiguous')
    lines = raw.raw_line.iloc[cs:de].to_numpy(); wanted = set(map(int, lines)); ratings = {}
    with gzip.open(source, 'rt', encoding='utf-8', newline='') as handle:
        for i, row in enumerate(csv.reader(handle)):
            if i in wanted:
                rating = float(row[2])
                if not np.isfinite(rating):
                    raise ValueError('Invalid C/D rating')
                ratings[i] = rating
    if len(ratings) != de-cs:
        raise ValueError('Missing raw C/D ratings')
    selected = pd.read_csv(processed, usecols=['ts', 'label'], skiprows=range(1, cs + 1), nrows=de-cs)
    if (len(selected) != de-cs or not np.array_equal(selected.ts.to_numpy(dtype=np.float64),
                                                     raw.ts.iloc[cs:de].to_numpy(dtype=np.float64))
            or not set(np.unique(selected.label)).issubset({0, 1})):
        raise ValueError('Invalid selected C/D labels')
    expected = np.array([ratings[int(i)] < 0 for i in lines], dtype=np.int8)
    labels = selected.label.to_numpy(dtype=np.int8)
    groups = {}
    for name, start, stop in (('reference_C', 0, ce-cs), ('development_D', ds-cs, de-cs)):
        groups[name] = {'events': stop-start, 'raw_negative': int(expected[start:stop].sum()),
                        'local_positive': int(labels[start:stop].sum()),
                        'label_mismatches': int(np.count_nonzero(expected[start:stop] != labels[start:stop]))}
    for policy in ('val_auroc', 'train_loss'):
        for run in runs:
            name = f"{policy}_seed{run['seed']}_scores.npz"; path = dev / name
            if path.is_symlink() or file_sha256(path) != seal['artifact_sha256'][name]:
                raise ValueError('Score artifact changed')
            with np.load(path, allow_pickle=False) as z:
                if (not np.array_equal(z['reference_times'], raw.ts.iloc[cs:ce].to_numpy(dtype=np.float64))
                        or not np.array_equal(z['development_times'], raw.ts.iloc[ds:de].to_numpy(dtype=np.float64))):
                    raise ValueError('Score timestamps differ from source events')
    edge = np.load(edge_path, mmap_mode='r', allow_pickle=False)
    node = np.load(node_path, mmap_mode='r', allow_pickle=False)
    if edge.ndim != 2 or node.ndim != 2 or edge.shape[0] <= de:
        raise ValueError('Feature dimensions do not cover audited events')
    feature_stats = {'node_shape': list(node.shape), 'node_dtype': str(node.dtype),
                     'node_nonzero': int(np.count_nonzero(node)), 'node_finite': bool(np.isfinite(node).all()),
                     'edge_shape': list(edge.shape), 'edge_dtype': str(edge.dtype)}
    for name, start, stop in (('train_and_selection', 0, cs), ('reference_C', cs, ce), ('development_D', ds, de)):
        values = edge[start+1:stop+1]
        feature_stats[name] = {'events': len(values), 'nonzero_elements': int(np.count_nonzero(values)),
                               'all_finite': bool(np.isfinite(values).all())}
    _verify_run(project, RUN)
    if (file_sha256(source) != source_hash or file_sha256(processed) != processed_hash
            or file_sha256(sp) != seal_hash or file_sha256(pp) != seal['protocol_sha256']):
        raise RuntimeError('Input changed during audit')
    return {'protocol': 'alpha_cd_label_provenance_audit_v1', 'run_id': RUN,
            'source_sha256': source_hash, 'processed_sha256': processed_hash,
            'development_seal_sha256': seal_hash, 'audit_code_sha256': file_sha256(Path(__file__).resolve()),
            'source_rows': len(raw), 'node_mapping_count': len(forward), 'C_bounds': [cs, ce],
            'D_bounds': [ds, de], 'old_test_start': test_start, 'groups': groups,
            'features': feature_stats, 'checked_score_files': 10,
            'limitations': 'raw score files lack event IDs; timestamp matches do not prove same-time event alignment; label release unknown; no numeric old Test ratings or labels'}


if __name__ == '__main__':
    print(json.dumps(audit(Path(__file__).resolve().parent), indent=2, allow_nan=False))
