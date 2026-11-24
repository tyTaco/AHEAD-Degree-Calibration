"""One-shot CPU development comparison; never reads old Test labels.

Requires a sealed shared-trajectory model trained with val_ratio=.05,
test_ratio=.5 and original degree calibration. This explicitly evaluates
offline, batch-delayed model scores: it is NOT an event-time causal protocol.
The ECDF candidate and strength are frozen before development evaluation.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from degree_development import batch_completion_times, chronological_development_split, paired_development_scores
from evaluate_shared_research import load_verified_shared
from train_research import build_research_model, file_sha256, set_research_seed
from utils.DataLoader import get_idx_data_loader
from utils.utils import get_neighbor_sampler

PRIOR_STRENGTH = 50.0  # Fixed synthetic-pilot candidate, not tuned on development.


def _pad_features(path):
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.shape[1] > 172:
        raise ValueError('Feature matrix must have at most 172 columns')
    if values.shape[1] < 172:
        values = np.concatenate([values, np.zeros((len(values), 172 - values.shape[1]))], axis=1)
    return values


def _data(frame):
    """Metadata only: no labels are attached to replay/reference data."""
    return SimpleNamespace(src_node_ids=frame.u.to_numpy(dtype=np.int64),
                           dst_node_ids=frame.i.to_numpy(dtype=np.int64),
                           node_interact_times=frame.ts.to_numpy(dtype=np.float64),
                           edge_ids=frame.idx.to_numpy(dtype=np.int64))


def collect_scores_and_degrees(model, sampler, loader, data, num_neighbors, time_gap):
    """Score once, recording degree from the same call; never read labels."""
    model.set_neighbor_sampler(sampler)
    model.eval()
    original = model._get_temporal_degree_from_sampler
    recorded = []
    def capture(*args, **kwargs):
        result = np.asarray(original(*args, **kwargs)).reshape(-1).copy()
        recorded.append(result)
        return result
    model._get_temporal_degree_from_sampler = capture
    scores = []
    try:
        with torch.no_grad():
            for indexes in loader:
                idx = indexes.numpy()
                before = len(recorded)
                result = model.compute_anomaly_score(
                    src_node_ids=data.src_node_ids[idx], dst_node_ids=data.dst_node_ids[idx],
                    node_interact_times=data.node_interact_times[idx], edge_ids=data.edge_ids[idx],
                    num_neighbors=num_neighbors, time_gap=time_gap)
                if (not isinstance(result, torch.Tensor) or result.numel() != len(idx)
                        or not torch.isfinite(result).all().item() or len(recorded) != before + 1
                        or len(recorded[-1]) != len(idx)):
                    raise ValueError('Scores or recorded degrees do not align with events')
                scores.append(result.detach().cpu().reshape(-1).numpy())
    finally:
        model._get_temporal_degree_from_sampler = original
    if not scores:
        raise ValueError('Empty scoring segment')
    return np.concatenate(scores), np.concatenate(recorded)


def run_development(project_dir, run_id, confirm_development):
    if not confirm_development:
        raise ValueError('Explicit --confirm-development is required')
    project = Path(project_dir).resolve()
    if Path.cwd().resolve() != project:
        raise ValueError('Run from the src directory')
    root, provenance, runs = load_verified_shared(project, run_id)
    config = SimpleNamespace(**provenance['configuration'])
    if (config.val_ratio != .05 or config.test_ratio != .5 or config.num_neighbors != 20
            or config.sample_neighbor_strategy != 'recent' or provenance['calibration_mode'] != 'original'):
        raise ValueError('Run does not match the predeclared development protocol')
    destination = root / 'degree_development'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Development attempt already exists; refusing to overwrite or repeat')

    files = {name: project / name for name in
             ('run_degree_development.py', 'degree_development.py', 'degree_ecdf_prototype.py')}
    code_hashes = {name: file_sha256(path) for name, path in files.items()}
    dataset = provenance['dataset']
    csv_path = project / 'processed_data' / dataset / f'ml_{dataset}.csv'
    # Old Test metadata is read to reproduce quantile boundaries. Its label
    # column is never requested, and it is excluded from the sampler.
    metadata = pd.read_csv(csv_path, usecols=['u', 'i', 'ts', 'idx'])
    times = metadata.ts.to_numpy(dtype=np.float64)
    parts = chronological_development_split(times)
    if parts['selection'].stop != np.searchsorted(times, np.quantile(times, 1 - config.test_ratio), side='right'):
        raise ValueError('Selection boundary differs from training')
    # Only development labels are requested; C and old Test labels are not read.
    dev = parts['development']
    labels = pd.read_csv(csv_path, usecols=['label'], skiprows=range(1, dev.start + 1),
                         nrows=dev.stop - dev.start)['label'].to_numpy()
    if len(labels) != dev.stop - dev.start or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Development labels must contain both binary classes')
    nodes = _pad_features(project / 'processed_data' / dataset / f'ml_{dataset}_node.npy')
    edges = _pad_features(project / 'processed_data' / dataset / f'ml_{dataset}.npy')
    prefix = _data(metadata.iloc[:dev.stop])
    selection = _data(metadata.iloc[parts['selection']])
    reference = _data(metadata.iloc[parts['reference']])
    development = _data(metadata.iloc[dev])
    model_ready_time = selection.node_interact_times[-1].item()
    ref_time, dev_time = reference.node_interact_times, development.node_interact_times
    if not model_ready_time < ref_time.min() <= ref_time.max() < dev_time.min():
        raise ValueError('Training/selection/reference/development time boundaries overlap')
    ref_available = batch_completion_times(ref_time, config.batch_size)
    dev_available = batch_completion_times(dev_time, config.batch_size)

    destination.mkdir(exist_ok=False)
    protocol = {'schema_version': 1, 'run_id': run_id, 'dataset': dataset,
                'input_provenance_sha256': file_sha256(root / 'provenance.json'),
                'selection_seal_sha256': file_sha256(root / 'selection_complete.json'),
                'development_code_sha256': code_hashes,
                'segments': {name: [part.start, part.stop] for name, part in parts.items()},
                'model_ready_time': model_ready_time, 'degree_cap': 20, 'prior_strength': PRIOR_STRENGTH,
                'batch_size': config.batch_size,
                'last_batch_sizes': {name: ((len(segment.src_node_ids) - 1) % config.batch_size) + 1
                                     for name, segment in (('selection', selection), ('reference', reference),
                                                           ('development', development))},
                'scoring_protocol': 'offline_batch_delayed_within_batch_future_candidates',
                'score_time_semantics': 'last_event_timestamp_in_each_batch',
                'old_test_labels_read': False,
                'reference_labels_read': False, 'development_labels_used_for_metrics': True,
                'selection_policies': ['val_auroc', 'train_loss'],
                'methods': ['none', 'original', 'continuous', 'degree_ecdf_prototype']}
    with (destination / 'development_started.json').open('x') as handle:
        json.dump(protocol, handle, indent=2, allow_nan=False)
    old_mode = os.environ.get('AHEAD_DEGREE_CALIB_MODE')
    os.environ['AHEAD_DEGREE_CALIB_MODE'] = 'none'
    try:
        config.device = 'cpu'
        results = {'val_auroc': [], 'train_loss': []}
        artifacts = {}
        for run in runs:
            seed = run['seed']
            for policy in ('val_auroc', 'train_loss'):
                set_research_seed(seed, 'cpu')
                sampler = get_neighbor_sampler(prefix, config.sample_neighbor_strategy, config.time_scaling_factor, seed=1)
                model = build_research_model(config, nodes, edges, sampler, seed).to('cpu')
                chosen = run['policies'][policy]
                checkpoint = root / f'seed{seed}' / chosen['checkpoint']
                if file_sha256(checkpoint) != chosen['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed before scoring')
                model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
                model.memory_generator.grad_mask_dict = None
                def loader(segment):
                    return get_idx_data_loader(list(range(len(segment.src_node_ids))), config.batch_size, False)
                # Replay B to align selected post-Train memory with reference C.
                collect_scores_and_degrees(model, sampler, loader(selection), selection,
                                           config.num_neighbors, config.time_gap)
                ref_scores, ref_degrees = collect_scores_and_degrees(model, sampler, loader(reference), reference,
                                                                      config.num_neighbors, config.time_gap)
                raw, degrees = collect_scores_and_degrees(model, sampler, loader(development), development,
                                                           config.num_neighbors, config.time_gap)
                values, metrics = paired_development_scores(
                    ref_scores, ref_degrees, ref_available, raw, degrees, dev_available, labels,
                    model_ready_time=model_ready_time, prior_strength=PRIOR_STRENGTH)
                item = {'seed': seed, 'policy': policy, 'selected_epoch': chosen['epoch'],
                        'checkpoint_sha256': chosen['checkpoint_sha256'],
                        'validation_labels_used_for_selection': policy == 'val_auroc', 'metrics': metrics}
                metrics_path = destination / f'{policy}_seed{seed}_metrics.json'
                scores_path = destination / f'{policy}_seed{seed}_scores.npz'
                with metrics_path.open('x') as handle:
                    json.dump(item, handle, indent=2, allow_nan=False)
                with scores_path.open('xb') as handle:
                    np.savez_compressed(handle, reference_scores=ref_scores, reference_degrees=ref_degrees,
                                        reference_times=ref_time, reference_score_available_at=ref_available,
                                        development_degrees=degrees, development_times=dev_time,
                                        development_score_available_at=dev_available, **values)
                for path in (metrics_path, scores_path):
                    artifacts[path.name] = file_sha256(path)
                results[policy].append(item)
                print(f'Development scoring complete: {policy}, seed {seed}', flush=True)
        if any(file_sha256(path) != code_hashes[name] for name, path in files.items()):
            raise ValueError('Development code changed during scoring; refusing to seal')
        load_verified_shared(project, run_id)
        if any(file_sha256(destination / name) != digest for name, digest in artifacts.items()):
            raise ValueError('Development artifacts changed before sealing')
        summary, paired_deltas = {}, {}
        for policy, items in results.items():
            summary[policy] = {}
            paired_deltas[policy] = {}
            for method in protocol['methods']:
                summary[policy][method] = {}
                paired_deltas[policy][method] = {}
                for metric in ('AUROC', 'AP', 'PR_AUC'):
                    vals = np.array([item['metrics'][method][metric] for item in items])
                    summary[policy][method][metric] = {'mean': float(vals.mean()),
                                                        'sample_sd': float(vals.std(ddof=1)) if len(vals) > 1 else None}
                    base = np.array([item['metrics']['none'][metric] for item in items])
                    delta = vals - base
                    paired_deltas[policy][method][metric] = {'mean': float(delta.mean()),
                                                              'sample_sd': float(delta.std(ddof=1)) if len(delta) > 1 else None}
        with (destination / 'development_complete.json').open('x') as handle:
            json.dump({'num_runs': len(runs), 'summary': summary, 'paired_delta_vs_none': paired_deltas,
                       'protocol_sha256': file_sha256(destination / 'development_started.json'),
                       'artifact_sha256': artifacts},
                      handle, indent=2, allow_nan=False)
        return summary
    finally:
        if old_mode is None:
            os.environ.pop('AHEAD_DEGREE_CALIB_MODE', None)
        else:
            os.environ['AHEAD_DEGREE_CALIB_MODE'] = old_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--confirm-development', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_development(Path(__file__).resolve().parent, args.run_id, args.confirm_development), indent=2))


if __name__ == '__main__':
    main()
