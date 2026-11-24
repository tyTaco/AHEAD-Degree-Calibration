"""Convert a locally supplied JODIE-format CSV to a new AHEAD data directory.

This tool never downloads. It preserves the original event order and labels;
source and destination IDs are mapped to disjoint positive integer ranges.
ID mappings and feature matrix dimensions use the full input (transductive).
It does not evaluate, select models or compute label-based statistics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source, destination, dataset):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if (not isinstance(dataset, str) or dataset in ('', '.', '..')
            or Path(dataset).name != dataset or not dataset.isidentifier()):
        raise ValueError('Dataset name must be a simple identifier')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Destination already exists; refusing to overwrite')
    # All label values must survive conversion, but are not used for mappings,
    # filtering, tuning, metrics, or selection.
    before = sha256(source)
    with source.open('r', encoding='utf-8', newline='') as stream:
        header = next(csv.reader(stream), [])
    if (len(header) < 5 or header[0] not in ('user', 'user_id')
            or header[1] not in ('item', 'item_id') or header[2:4] != ['timestamp', 'state_label']):
        raise ValueError('Expected JODIE header user,item,timestamp,state_label and features')
    # JODIE header has one feature-list placeholder even when each row has
    # many feature columns, so the header must not be used as a pandas schema.
    frame = pd.read_csv(source, skiprows=1, header=None, dtype={0: 'string', 1: 'string'})
    if len(frame.columns) < 5 or frame.empty:
        raise ValueError('Expected at least one feature per interaction')
    if frame.iloc[:, :4].isna().any().any():
        raise ValueError('IDs, timestamps and labels must be present')
    times = pd.to_numeric(frame.iloc[:, 2], errors='raise').to_numpy(dtype=np.float64)
    if (not np.isfinite(times).all() or np.any(np.abs(times) >= 2**53)
            or np.any(np.diff(times) < 0)):
        raise ValueError('Timestamps must be finite, precise and chronological')
    labels = pd.to_numeric(frame.iloc[:, 3], errors='raise').to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary state labels')
    features = frame.iloc[:, 4:].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if features.shape[1] > 172 or not np.isfinite(features).all():
        raise ValueError('Expected finite edge features with at most 172 dimensions')
    src, users = pd.factorize(frame.iloc[:, 0], sort=False)
    dst, items = pd.factorize(frame.iloc[:, 1], sort=False)
    graph = pd.DataFrame({'u': src + 1, 'i': dst + 1 + len(users), 'ts': times,
                          'label': labels.astype(np.int8), 'idx': np.arange(1, len(frame) + 1)})
    node_features = np.zeros((len(users) + len(items) + 1, 172), dtype=np.float32)
    edge_features = np.concatenate([np.zeros((1, features.shape[1]), dtype=np.float32), features])
    if sha256(source) != before:
        raise RuntimeError('Source changed during conversion')
    destination.mkdir(parents=True, exist_ok=False)
    paths = {'csv': destination / f'ml_{dataset}.csv', 'edge_features': destination / f'ml_{dataset}.npy',
             'node_features': destination / f'ml_{dataset}_node.npy'}
    with paths['csv'].open('x', encoding='utf-8', newline='') as stream:
        graph.to_csv(stream, index=False)
    for key, values in [('edge_features', edge_features), ('node_features', node_features)]:
        with paths[key].open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
    manifest = {'schema_version': 1, 'source_sha256': before, 'source_name': source.name,
                'converter_sha256': sha256(Path(__file__).resolve()), 'event_count': len(frame),
                'source_node_count': len(users), 'destination_node_count': len(items),
                'edge_feature_dim': features.shape[1], 'id_mapping': 'first_occurrence_disjoint_bipartite',
                'edge_feature_dtype': 'float32',
                'labels': 'preserved_from_state_label', 'all_input_metadata_used_for_id_mapping': True,
                'artifacts': {p.name: sha256(p) for p in paths.values()}}
    with (destination / 'conversion_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.dataset), indent=2))


if __name__ == '__main__':
    main()
