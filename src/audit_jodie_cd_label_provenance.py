"""Read-only Wikipedia/MOOC C/D label provenance and feature audit.

The raw CSV is scanned for all user/item/time metadata, including later events,
to check alignment and the last observed interaction. Only C/D labels are
converted to numbers. The raw label bytes are still read; hash/CSV buffering
can access other bytes. Last observed is not proof of true label release time.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from run_label_gate_development import _verify_run
from train_research import file_sha256

RUNS = {'wikipedia': ('wikipedia_shared_dev_5seed_fixed100_20261121', 'raw_data/wikipedia.csv'),
        'mooc': ('mooc_segmented_dev_5seed_fixed100_20261121', 'raw_data/mooc_jodie_20190714.csv')}


def audit(project, dataset):
    if dataset not in RUNS:
        raise ValueError('Only audited Wikipedia and MOOC runs are supported')
    project = Path(project).resolve()
    run_id, raw_name = RUNS[dataset]
    root, provenance, runs = _verify_run(project, run_id)
    if provenance['dataset'] != dataset or sorted(x['seed'] for x in runs) != list(range(5)):
        raise ValueError('Unexpected dataset or seeds')
    dev = root / 'degree_development'
    pp, sp = dev / 'development_started.json', dev / 'development_complete.json'
    if any(p.is_symlink() for p in (dev, pp, sp)):
        raise ValueError('Redirected development artifacts')
    protocol, seal = json.loads(pp.read_text()), json.loads(sp.read_text()); seal_hash = file_sha256(sp)
    if (file_sha256(pp) != seal['protocol_sha256'] or protocol['dataset'] != dataset
            or protocol['run_id'] != run_id or seal['num_runs'] != 5):
        raise ValueError('Development provenance mismatch')
    cs, ce = protocol['segments']['reference']; ds, de = protocol['segments']['development']
    test_start, _ = protocol['segments']['old_test']
    if not 0 < cs < ce == ds < de <= test_start:
        raise ValueError('C/D overlaps old Test')
    source = project / raw_name
    base = project / 'processed_data' / dataset
    processed = base / f'ml_{dataset}.csv'; edge_path = base / f'ml_{dataset}.npy'; node_path = base / f'ml_{dataset}_node.npy'
    if any(x.is_symlink() or not x.is_file() for x in (source, processed, edge_path, node_path)):
        raise ValueError('Missing or redirected source or processed input')
    source_hash, processed_hash = file_sha256(source), file_sha256(processed)
    metadata = pd.read_csv(processed, usecols=['u', 'i', 'ts', 'idx'])
    if len(metadata) < de or not np.array_equal(metadata.idx.to_numpy(), np.arange(1, len(metadata) + 1)):
        raise ValueError('Processed edge IDs are not sequential')
    times = metadata.ts.to_numpy(dtype=np.float64)
    source_ids = metadata.u.to_numpy(dtype=np.int64)
    target_ids = metadata.i.to_numpy(dtype=np.int64)
    if (not np.isfinite(times).all() or np.any(times[1:] < times[:-1])
            or not times[cs-1] < times[cs] or not times[ce-1] < times[ce]
            or (de < len(times) and not times[de-1] < times[de])):
        raise ValueError('Segment times are not strictly separated')
    labels = pd.read_csv(processed, usecols=['ts', 'label'], skiprows=range(1, cs + 1), nrows=de-cs)
    if (len(labels) != de-cs or not set(np.unique(labels.label)).issubset({0, 1})
            or not np.array_equal(labels.ts.to_numpy(dtype=np.float64), times[cs:de])):
        raise ValueError('Invalid processed C/D labels')
    users, users_reverse, items, items_reverse = {}, {}, {}, {}
    last_user, chosen = {}, []
    row_count = 0
    with source.open('rb') as raw:
        header = raw.readline().strip().split(b',')
        if len(header) < 5 or header[0] not in (b'user', b'user_id') or header[1] not in (b'item', b'item_id') or header[2:4] != [b'timestamp', b'state_label']:
            raise ValueError('Unexpected JODIE header')
        for index, line in enumerate(raw):
            fields = line.rstrip(b'\r\n').split(b',', 4)
            if len(fields) != 5 or index >= len(metadata):
                raise ValueError('Unexpected raw event row')
            user, item, raw_time, label_text, _ = fields
            timestamp = float(raw_time)
            if not np.isfinite(timestamp) or timestamp != times[index]:
                raise ValueError('Raw and processed timestamps differ')
            for original, local, forward, reverse in ((user, int(source_ids[index]), users, users_reverse),
                                                       (item, int(target_ids[index]), items, items_reverse)):
                if (original in forward and forward[original] != local) or (local in reverse and reverse[local] != original):
                    raise ValueError('Raw and processed endpoint mapping differs')
                forward[original], reverse[local] = local, original
            last_user[user] = index
            if cs <= index < de:
                if label_text not in (b'0', b'1', b'0.0', b'1.0'):
                    raise ValueError('C/D raw label is not binary')
                value = 1 if label_text in (b'1', b'1.0') else 0
                chosen.append((index, user, item, timestamp, value))
            row_count += 1
    if row_count != len(metadata) or len(chosen) != de-cs:
        raise ValueError('Raw and processed event counts differ')
    keys = Counter((u, i, t) for _, u, i, t, _ in chosen)
    raw_classes, local_classes = {}, {}
    raw_labels = np.asarray([value for _, _, _, _, value in chosen], dtype=np.int8)
    saved_labels = labels.label.to_numpy(dtype=np.int8)
    for offset, (_, user, item, timestamp, value) in enumerate(chosen):
        key = (user, item, timestamp)
        if keys[key] > 1:
            raw_classes.setdefault(key, set()).add(value)
            local_classes.setdefault(key, set()).add(int(saved_labels[offset]))
    groups = {}
    for name, start, stop in (('reference_C', 0, ce-cs), ('development_D', ds-cs, de-cs)):
        section = chosen[start:stop]
        positives = [(idx, user) for idx, user, _, _, value in section if value == 1]
        duplicate_keys = {(u, i, t) for _, u, i, t, _ in section if keys[(u, i, t)] > 1}
        raw_conflicts = {key for key in duplicate_keys if len(raw_classes[key]) > 1}
        local_conflicts = {key for key in duplicate_keys if len(local_classes[key]) > 1}
        groups[name] = {'events': stop-start, 'raw_positive': int(raw_labels[start:stop].sum()),
            'local_positive': int(saved_labels[start:stop].sum()),
            'label_mismatches': int(np.count_nonzero(raw_labels[start:stop] != saved_labels[start:stop])),
            'raw_positive_is_last_observed_user_interaction': sum(last_user[user] == idx for idx, user in positives),
            'duplicate_source_target_time_rows': sum(keys[(u, i, t)] > 1 for _, u, i, t, _ in section),
            'duplicate_key_groups': len(duplicate_keys),
            'raw_duplicate_label_conflict_groups': len(raw_conflicts),
            'local_duplicate_label_conflict_groups': len(local_conflicts)}
    for policy in ('val_auroc', 'train_loss'):
        for run in runs:
            name = f"{policy}_seed{run['seed']}_scores.npz"; path = dev / name
            if path.is_symlink() or file_sha256(path) != seal['artifact_sha256'][name]:
                raise ValueError('Score artifact changed')
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z['reference_times'], times[cs:ce]) or not np.array_equal(z['development_times'], times[ds:de]):
                    raise ValueError('Score timestamp alignment mismatch')
    edges = np.load(edge_path, mmap_mode='r', allow_pickle=False)
    nodes = np.load(node_path, mmap_mode='r', allow_pickle=False)
    if edges.ndim != 2 or nodes.ndim != 2 or edges.shape[0] <= de:
        raise ValueError('Feature dimensions do not cover C/D')
    feature_stats = {'edge_shape': list(edges.shape), 'node_shape': list(nodes.shape),
                     'node_nonzero_elements': int(np.count_nonzero(nodes)), 'node_all_finite': bool(np.isfinite(nodes).all())}
    for name, start, stop in (('reference_C', cs, ce), ('development_D', ds, de)):
        features = edges[start+1:stop+1]
        y = saved_labels[start-cs:stop-cs]
        equal = np.flatnonzero(np.all(features == y[:, None], axis=0)).tolist()
        inverse = np.flatnonzero(np.all(features == (1-y)[:, None], axis=0)).tolist()
        feature_stats[name] = {'events': len(features), 'all_finite': bool(np.isfinite(features).all()),
                               'exact_label_columns': equal, 'exact_inverse_label_columns': inverse}
    _verify_run(project, run_id)
    if (file_sha256(source) != source_hash or file_sha256(processed) != processed_hash
            or file_sha256(sp) != seal_hash or file_sha256(pp) != seal['protocol_sha256']):
        raise RuntimeError('Inputs changed during audit')
    return {'protocol': 'jodie_cd_label_provenance_audit_v1', 'dataset': dataset, 'run_id': run_id,
            'source_sha256': source_hash, 'processed_sha256': processed_hash, 'development_seal_sha256': seal_hash,
            'audit_code_sha256': file_sha256(Path(__file__).resolve()), 'source_rows': row_count,
            'users': len(users), 'items': len(items), 'C_bounds': [cs, ce], 'D_bounds': [ds, de],
            'old_test_start': test_start, 'groups': groups, 'features': feature_stats, 'checked_score_files': 10,
            'limitations': 'full user/item/time metadata used; old scores lack IDs; unknown label release; no numeric old Test labels; hash/CSV may read other bytes'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=tuple(RUNS))
    args = parser.parse_args()
    print(json.dumps(audit(Path(__file__).resolve().parent, args.dataset), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
