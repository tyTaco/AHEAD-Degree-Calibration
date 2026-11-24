"""Experimental shared-trajectory training with segmented label materialization.

Uses a new seal protocol. No Test evaluation is implemented here. Real runs
require user authorization and AHEAD_SEGMENTED_TRAINING_CONFIRM=1.
"""
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score

from segmented_selection_data import load_selection_data, model_data
from train_research import (build_research_model, collect_research_scores, file_sha256,
                            research_input_files, research_output_path, set_research_seed)
from train_shared_research import run_selection, run_train_epoch
from utils.DataLoader import get_idx_data_loader
from utils.load_configs import get_node_classification_args
from utils.utils import create_optimizer, get_neighbor_sampler

PROTOCOL = 'segmented_shared_trajectory_v1'


def input_files(project, dataset):
    files = research_input_files(project, dataset)
    for name in ('train_shared_research.py', 'segmented_selection_data.py',
                 'train_segmented_research.py', 'verify_segmented_research.py',
                 'run_segmented_development.py', 'degree_development.py', 'degree_ecdf_prototype.py'):
        files[name] = Path(project) / name
    manifest = Path(project) / 'processed_data' / dataset / 'conversion_manifest.json'
    if manifest.exists() or manifest.is_symlink():
        files[str(manifest.relative_to(project))] = manifest
        files['prepare_jodie_research.py'] = Path(project) / 'prepare_jodie_research.py'
    return files


def pad_features(path):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] > 172 or not np.isfinite(values).all():
        raise ValueError('Expected finite two-dimensional features with at most 172 columns')
    if values.shape[1] < 172:
        values = np.concatenate([values, np.zeros((len(values), 172 - values.shape[1]))], axis=1)
    return values


def verify_conversion(project, dataset):
    directory = Path(project) / 'processed_data' / dataset
    manifest = directory / 'conversion_manifest.json'
    if not manifest.exists() and not manifest.is_symlink():
        return
    if manifest.is_symlink():
        raise ValueError('Conversion manifest must not be a symlink')
    record = json.loads(manifest.read_text())
    expected = {f'ml_{dataset}.csv', f'ml_{dataset}.npy', f'ml_{dataset}_node.npy'}
    artifacts = record.get('artifacts', {})
    if (record.get('schema_version') != 1 or set(artifacts) != expected
            or record.get('converter_sha256') != file_sha256(Path(project) / 'prepare_jodie_research.py')):
        raise ValueError('Conversion manifest or converter mismatch')
    for name in expected:
        path = directory / name
        if path.is_symlink() or file_sha256(path) != artifacts[name]:
            raise ValueError('Converted data mismatch')


def run(project, args, run_id, mode, confirmed):
    if not confirmed:
        raise ValueError('Explicit authorization is required for training')
    project = Path(project).resolve()
    if Path.cwd().resolve() != project:
        raise ValueError('Run from the src directory')
    if (not isinstance(args.dataset_name, str) or args.dataset_name in ('', '.', '..')
            or Path(args.dataset_name).name != args.dataset_name):
        raise ValueError('Invalid dataset name')
    if (args.model_name != 'AHEADGraphMixer' or min(args.num_runs, args.num_epochs, args.batch_size) < 1
            or args.sample_neighbor_strategy != 'recent' or mode not in ('original', 'none', 'continuous')):
        raise ValueError('Invalid model, run settings, sampler, or calibration')
    verify_conversion(project, args.dataset_name)
    files = input_files(project, args.dataset_name)
    fingerprints = {key: {'sha256': file_sha256(path), 'bytes': path.stat().st_size}
                    for key, path in files.items()}
    split = load_selection_data(project / 'processed_data' / args.dataset_name / f'ml_{args.dataset_name}.csv',
                                val_ratio=args.val_ratio, test_ratio=args.test_ratio, use_selection_labels=True)
    train, val, prefix = model_data(split)
    nodes = pad_features(project / 'processed_data' / args.dataset_name / f'ml_{args.dataset_name}_node.npy')
    edges = pad_features(project / 'processed_data' / args.dataset_name / f'ml_{args.dataset_name}.npy')
    if (min(prefix.src_node_ids.min(), prefix.dst_node_ids.min(), prefix.edge_ids.min()) < 1
            or max(prefix.src_node_ids.max(), prefix.dst_node_ids.max()) >= len(nodes)
            or prefix.edge_ids.max() >= len(edges)):
        raise ValueError('Prefix IDs must be positive and fit the feature matrices')
    train_sampler = get_neighbor_sampler(train, 'recent', args.time_scaling_factor, seed=0)
    val_sampler = get_neighbor_sampler(prefix, 'recent', args.time_scaling_factor, seed=1)
    train_loader = get_idx_data_loader(list(range(train.num_interactions)), args.batch_size, False)
    val_loader = get_idx_data_loader(list(range(val.num_interactions)), args.batch_size, False)
    root = research_output_path(project, run_id)
    root.mkdir(parents=True, exist_ok=False)
    provenance = {'schema_version': 1, 'protocol': PROTOCOL, 'dataset': args.dataset_name,
                  'calibration_mode': mode, 'configuration': vars(args).copy(),
                  'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                  'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'files': fingerprints,
                  'selection_label_materialization': 'B_only', 'future_metadata_used_for_quantiles': True,
                  'sampler_scope': 'training_or_A_plus_B', 'features': 'entire_feature_arrays'}
    with (root / 'provenance.json').open('x') as handle:
        json.dump(provenance, handle, indent=2, allow_nan=False)
    results = []
    previous = os.environ.get('AHEAD_DEGREE_CALIB_MODE')
    os.environ['AHEAD_DEGREE_CALIB_MODE'] = mode
    try:
        for seed in range(args.num_runs):
            set_research_seed(seed, args.device)
            train_sampler.reset_random_state()
            val_sampler.reset_random_state()
            model = build_research_model(args, nodes, edges, train_sampler, seed).to(args.device)
            optimizer = create_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
            args.train_sampler = train_sampler
            def train_one(epoch):
                return run_train_epoch(model, optimizer, train_loader, train, args, epoch)
            def validate():
                val_sampler.reset_random_state()
                scores, labels = collect_research_scores(model, val_sampler, val_loader, val,
                                                         args.num_neighbors, args.time_gap, True)
                return roc_auc_score(labels, scores)
            seed_dir = root / f'seed{seed}'
            record = run_selection(model, train_one, validate, args.num_epochs, seed_dir)
            del args.train_sampler
            results.append({'seed': seed, 'selection_sha256': file_sha256(seed_dir / 'selection.json'),
                            'policies': record['policies']})
    finally:
        if previous is None:
            os.environ.pop('AHEAD_DEGREE_CALIB_MODE', None)
        else:
            os.environ['AHEAD_DEGREE_CALIB_MODE'] = previous
        if hasattr(args, 'train_sampler'):
            del args.train_sampler
    for key, path in files.items():
        if file_sha256(path) != fingerprints[key]['sha256']:
            raise RuntimeError(f'Input changed during training: {key}')
    for result in results:
        seed_dir = root / f"seed{result['seed']}"
        if file_sha256(seed_dir / 'selection.json') != result['selection_sha256']:
            raise RuntimeError('Selection record changed')
        for selected in result['policies'].values():
            if file_sha256(seed_dir / selected['checkpoint']) != selected['checkpoint_sha256']:
                raise RuntimeError('Selected checkpoint changed')
    with (root / 'selection_complete.json').open('x') as handle:
        json.dump({'schema_version': 1, 'protocol': PROTOCOL,
                   'provenance_sha256': file_sha256(root / 'provenance.json'), 'runs': results},
                  handle, indent=2, allow_nan=False)
    return root


def main():
    if os.environ.get('AHEAD_SEGMENTED_TRAINING_CONFIRM') != '1':
        raise ValueError('Explicit training authorization is required')
    args = get_node_classification_args()
    root = run(Path(__file__).resolve().parent, args, os.environ.get('AHEAD_RESEARCH_RUN_ID', ''),
               os.environ.get('AHEAD_DEGREE_CALIB_MODE', 'original'), True)
    logging.info('Segmented selections sealed: %s; no Test evaluation performed.', root)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
