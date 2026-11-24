"""Read-only verification of the experimental segmented selection seal."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from train_research import file_sha256, research_output_path
from train_segmented_research import PROTOCOL, input_files, verify_conversion


def verify(project, run_id):
    project = Path(project).resolve()
    root = research_output_path(project, run_id)
    if not root.is_dir() or (root / 'final_test').exists() or (root / 'final_test').is_symlink():
        raise ValueError('Run missing or Test already started')
    provenance_path, seal_path = root / 'provenance.json', root / 'selection_complete.json'
    if provenance_path.is_symlink() or seal_path.is_symlink():
        raise ValueError('Redirected seal')
    provenance, seal = json.loads(provenance_path.read_text()), json.loads(seal_path.read_text())
    if (seal.get('schema_version') != 1 or seal.get('protocol') != PROTOCOL
            or provenance.get('protocol') != PROTOCOL or provenance.get('schema_version') != 1
            or file_sha256(provenance_path) != seal.get('provenance_sha256')):
        raise ValueError('Invalid seal or provenance')
    versions = {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}
    if any(provenance.get(key) != value for key, value in versions.items()):
        raise ValueError('Environment changed')
    dataset = provenance.get('dataset')
    if not isinstance(dataset, str) or dataset in ('', '.', '..') or Path(dataset).name != dataset:
        raise ValueError('Invalid dataset')
    verify_conversion(project, dataset)
    files = input_files(project, dataset)
    recorded = provenance.get('files', {})
    if set(files) != set(recorded):
        raise ValueError('Input file list changed')
    for key, path in files.items():
        if not path.is_file() or file_sha256(path) != recorded[key].get('sha256'):
            raise ValueError(f'Input changed: {key}')
    config, runs = provenance.get('configuration', {}), seal.get('runs')
    if (config.get('dataset_name') != dataset or config.get('model_name') != 'AHEADGraphMixer'
            or config.get('sample_neighbor_strategy') != 'recent'
            or provenance.get('calibration_mode') not in ('original', 'none', 'continuous')
            or provenance.get('selection_label_materialization') != 'B_only'
            or provenance.get('sampler_scope') != 'training_or_A_plus_B'
            or provenance.get('future_metadata_used_for_quantiles') is not True
            or provenance.get('features') != 'entire_feature_arrays'
            or not isinstance(runs, list) or not isinstance(config.get('num_runs'), int)
            or config['num_runs'] < 1 or not isinstance(config.get('num_epochs'), int)
            or config['num_epochs'] < 1 or len(runs) != config['num_runs']
            or [item.get('seed') for item in runs] != list(range(config['num_runs']))):
        raise ValueError('Invalid configuration or seeds')
    for run in runs:
        folder = root / f"seed{run['seed']}"
        if folder.is_symlink() or folder.resolve() != folder:
            raise ValueError('Redirected seed directory')
        record_path = folder / 'selection.json'
        if record_path.is_symlink() or file_sha256(record_path) != run.get('selection_sha256'):
            raise ValueError('Selection record changed')
        record = json.loads(record_path.read_text())
        history = record.get('history')
        if (record.get('fixed_epochs') != config.get('num_epochs')
                or record.get('policies') != run.get('policies')
                or set(run['policies']) != {'val_auroc', 'train_loss'}
                or not isinstance(history, list) or len(history) != config['num_epochs']
                or [x.get('epoch') for x in history] != list(range(1, config['num_epochs'] + 1))):
            raise ValueError('Invalid selection history')
        for policy, selected in run['policies'].items():
            scores = [x.get(policy) for x in history]
            if (any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in scores)
                    or (policy == 'val_auroc' and any(v < 0 or v > 1 for v in scores))):
                raise ValueError('Invalid history score')
            best = max(scores) if policy == 'val_auroc' else min(scores)
            epoch = scores.index(best) + 1
            checkpoint = folder / f'{policy}_epoch{epoch:04d}.pkl'
            if (selected.get('epoch') != epoch or selected.get('score') != best
                    or selected.get('checkpoint') != checkpoint.name or checkpoint.is_symlink()
                    or selected.get('validation_labels_used_for_selection') != (policy == 'val_auroc')
                    or not checkpoint.is_file() or file_sha256(checkpoint) != selected.get('checkpoint_sha256')):
                raise ValueError('Invalid selected checkpoint')
    return root, provenance, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    root, provenance, runs = verify(Path(__file__).resolve().parent, args.run_id)
    print(json.dumps({'run': str(root), 'protocol': provenance['protocol'], 'verified_seeds': len(runs)}))


if __name__ == '__main__':
    main()
