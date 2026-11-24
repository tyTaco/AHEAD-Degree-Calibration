"""One-shot CPU Test evaluator for sealed shared-trajectory selections."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
import torch

from train_research import (build_research_model, collect_research_scores, file_sha256,
                            research_input_files, research_output_path, research_ranking_metrics,
                            set_research_seed)
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.utils import get_neighbor_sampler


def load_verified_shared(project_dir, run_id):
    project = Path(project_dir).resolve()
    root = research_output_path(project, run_id)
    if not root.is_dir() or (root / 'final_test').exists() or (root / 'final_test').is_symlink():
        raise ValueError('Run is missing or has already started final Test')
    seal_path, prov_path = root / 'selection_complete.json', root / 'provenance.json'
    if seal_path.is_symlink() or prov_path.is_symlink():
        raise ValueError('Seals must not be symlinks')
    sealed, prov = json.loads(seal_path.read_text()), json.loads(prov_path.read_text())
    if sealed.get('schema_version') != 1 or sealed.get('protocol') != 'shared_trajectory_fixed_epochs':
        raise ValueError('Invalid selection seal')
    if prov.get('protocol') != sealed['protocol'] or file_sha256(prov_path) != sealed.get('provenance_sha256'):
        raise ValueError('Provenance changed or protocol mismatch')
    versions = {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                'pandas': pd.__version__, 'scikit_learn': sklearn.__version__}
    if any(prov.get(key) != value for key, value in versions.items()):
        raise ValueError('Environment version changed')
    dataset = prov.get('dataset')
    if not isinstance(dataset, str) or dataset in ('', '.', '..') or Path(dataset).name != dataset:
        raise ValueError('Invalid dataset')
    if prov.get('calibration_mode') not in ('original', 'none', 'continuous'):
        raise ValueError('Invalid calibration')
    files = research_input_files(project, dataset)
    files['train_shared_research.py'] = project / 'train_shared_research.py'
    files['evaluate_shared_research.py'] = project / 'evaluate_shared_research.py'
    recorded = prov.get('files', {})
    if set(files) != set(recorded):
        raise ValueError('Input file list changed')
    for key, path in files.items():
        if not path.is_file() or file_sha256(path) != recorded[key].get('sha256'):
            raise ValueError(f'Input changed: {key}')
    config = prov.get('configuration', {})
    runs = sealed.get('runs')
    if (config.get('dataset_name') != dataset or config.get('model_name') != 'AHEADGraphMixer'
            or not isinstance(runs, list) or not isinstance(config.get('num_runs'), int)
            or config['num_runs'] < 1 or len(runs) != config['num_runs']
            or [item.get('seed') for item in runs] != list(range(config['num_runs']))):
        raise ValueError('Missing or inconsistent seeds/configuration')
    for run in runs:
        folder = root / f"seed{run['seed']}"
        if folder.is_symlink() or folder.resolve() != folder:
            raise ValueError('Seed directory redirects outside run')
        record_path = folder / 'selection.json'
        if record_path.is_symlink() or file_sha256(record_path) != run.get('selection_sha256'):
            raise ValueError('Selection record changed')
        record = json.loads(record_path.read_text())
        if (record.get('fixed_epochs') != config.get('num_epochs') or record.get('policies') != run.get('policies')
                or set(run['policies']) != {'val_auroc', 'train_loss'}):
            raise ValueError('Selection record or fixed horizon mismatch')
        history = record.get('history')
        if (not isinstance(history, list) or len(history) != config['num_epochs']
                or [item.get('epoch') for item in history] != list(range(1, config['num_epochs'] + 1))):
            raise ValueError('Incomplete training history')
        for policy, selected in run['policies'].items():
            scores = [item.get(policy) for item in history]
            if any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in scores):
                raise ValueError('Invalid selection history')
            best_score = max(scores) if policy == 'val_auroc' else min(scores)
            if selected.get('score') != best_score or selected.get('epoch') != scores.index(best_score) + 1:
                raise ValueError('Selection does not match fixed-horizon history')
            epoch = selected.get('epoch')
            if not isinstance(epoch, int) or not 1 <= epoch <= config['num_epochs']:
                raise ValueError('Invalid selected epoch')
            expected = f'{policy}_epoch{epoch:04d}.pkl'
            checkpoint = folder / expected
            if (selected.get('checkpoint') != expected or checkpoint.is_symlink()
                    or selected.get('validation_labels_used_for_selection') != (policy == 'val_auroc')
                    or not checkpoint.is_file() or file_sha256(checkpoint) != selected.get('checkpoint_sha256')):
                raise ValueError('Selected checkpoint or policy mismatch')
    return root, prov, runs


def evaluate_shared(project_dir, run_id, confirm_test):
    if not confirm_test:
        raise ValueError('Explicit --confirm-test is required')
    project = Path(project_dir).resolve()
    if Path.cwd().resolve() != project:
        raise ValueError('Run from the src directory')
    root, prov, runs = load_verified_shared(project, run_id)
    mode = prov['calibration_mode']
    old_mode = os.environ.get('AHEAD_DEGREE_CALIB_MODE')
    if old_mode is not None and old_mode != mode:
        raise ValueError('Calibration environment conflicts with sealed run')
    out = root / 'final_test'
    out.mkdir(exist_ok=False)
    with (out / 'evaluation_started.json').open('x') as handle:
        json.dump({'device': 'cpu', 'calibration_mode': mode, 'num_runs': len(runs),
                   'memory_policy': 'selected_train_state_then_replay_validation_without_labels',
                   'evaluation_mask': 'none'}, handle, indent=2)
    os.environ['AHEAD_DEGREE_CALIB_MODE'] = mode
    try:
        config = SimpleNamespace(**prov['configuration'])
        config.device = 'cpu'
        nodes, edges, full, _train, val, test = get_node_classification_data(prov['dataset'], config.val_ratio, config.test_ratio)
        if np.any(np.diff(full.node_interact_times) < 0):
            raise ValueError('Interactions must be chronological')
        val_loader = get_idx_data_loader(list(range(len(val.src_node_ids))), config.batch_size, False)
        test_loader = get_idx_data_loader(list(range(len(test.src_node_ids))), config.batch_size, False)
        results = {'val_auroc': [], 'train_loss': []}
        reference_labels = None
        for run in runs:
            for policy in ('val_auroc', 'train_loss'):
                seed = run['seed']
                selected = run['policies'][policy]
                set_research_seed(seed, 'cpu')
                sampler = get_neighbor_sampler(full, config.sample_neighbor_strategy, config.time_scaling_factor, seed=1)
                model = build_research_model(config, nodes, edges, sampler, seed).to('cpu')
                path = root / f'seed{seed}' / selected['checkpoint']
                if file_sha256(path) != selected['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed before evaluation')
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
                model.memory_generator.grad_mask_dict = None
                collect_research_scores(model, sampler, val_loader, val, config.num_neighbors, config.time_gap, False)
                scores, labels = collect_research_scores(model, sampler, test_loader, test,
                                                         config.num_neighbors, config.time_gap, True)
                if reference_labels is None:
                    reference_labels = labels.copy()
                    with (out / 'labels.npy').open('xb') as handle:
                        np.save(handle, labels)
                elif not np.array_equal(reference_labels, labels):
                    raise ValueError('Test labels differ between selections')
                metrics = research_ranking_metrics(labels, scores)
                item = {'seed': seed, 'policy': policy, 'selected_epoch': selected['epoch'],
                        'checkpoint_sha256': selected['checkpoint_sha256'], 'metrics': metrics}
                with (out / f'{policy}_seed{seed}_metrics.json').open('x') as handle:
                    json.dump(item, handle, indent=2, allow_nan=False)
                with (out / f'{policy}_seed{seed}_scores.npy').open('xb') as handle:
                    np.save(handle, scores)
                results[policy].append(metrics)
        summary = {}
        for policy, entries in results.items():
            summary[policy] = {}
            for metric in ('AUROC', 'AP', 'PR_AUC'):
                values = np.array([entry[metric] for entry in entries])
                summary[policy][metric] = {'mean': float(values.mean()),
                                           'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else None}
        with (out / 'evaluation_complete.json').open('x') as handle:
            json.dump({'num_runs': len(runs), 'summary': summary}, handle, indent=2, allow_nan=False)
        return summary
    finally:
        if old_mode is None:
            os.environ.pop('AHEAD_DEGREE_CALIB_MODE', None)
        else:
            os.environ['AHEAD_DEGREE_CALIB_MODE'] = old_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--confirm-test', action='store_true')
    args = parser.parse_args()
    print(json.dumps(evaluate_shared(Path(__file__).resolve().parent, args.run_id, args.confirm_test), indent=2))


if __name__ == '__main__':
    main()
