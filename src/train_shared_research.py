"""Shared-trajectory AHEAD selection. This entry never evaluates Test.

Both policies select from the same fixed number of epochs. Validation labels are
used only for the supervised selection record, not stopping or gradient updates.
Real training requires explicit authorization in addition to the opt-in flag.
"""
import copy
import json
import logging
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from train_research import (build_research_model, collect_research_scores, file_sha256,
                            research_input_files, research_output_path, set_research_seed)
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.load_configs import get_node_classification_args
from utils.utils import create_optimizer, get_neighbor_sampler


@contextmanager
def isolated_validation(model, device):
    """Restore training state, RNG streams, mode, mask, and sampler references."""
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    modes = [(module, module.training) for module in model.modules()]
    mask = copy.deepcopy(model.memory_generator.grad_mask_dict)
    model_sampler = model.neighbor_sampler
    generator_sampler = model.memory_generator.neighbor_sampler
    py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.device(device).type == 'cuda' else None
    try:
        model.memory_generator.grad_mask_dict = None
        yield
    finally:
        model.load_state_dict(state, strict=True)
        model.memory_generator.grad_mask_dict = mask
        # Avoid set_neighbor_sampler: it can reset the training sampler's RNG.
        model.neighbor_sampler = model_sampler
        model.memory_generator.neighbor_sampler = generator_sampler
        for module, mode in modes:
            module.training = mode
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def run_train_epoch(model, optimizer, loader, data, args, epoch):
    """Reuse the research training stages; no Validation or Test labels here."""
    model.eval()
    model.reset_states()
    model.set_neighbor_sampler(args.train_sampler)
    model.memory.reset_memory()
    for indexes in tqdm(loader, desc=f'Epoch {epoch} warm-up', leave=False):
        idx = indexes.numpy()
        with torch.no_grad():
            model.compute_anomaly_score(data.src_node_ids[idx], data.dst_node_ids[idx],
                                        data.node_interact_times[idx], data.edge_ids[idx],
                                        num_neighbors=args.num_neighbors, time_gap=args.time_gap)
    model.memory.reset_memory()
    for indexes in tqdm(loader, desc=f'Epoch {epoch} indicators', leave=False):
        idx = indexes.numpy()
        model.compute_and_cache_indicators(data.src_node_ids[idx], data.dst_node_ids[idx],
                                           data.node_interact_times[idx], data.edge_ids[idx],
                                           num_neighbors=args.num_neighbors, time_gap=args.time_gap)
    model.indicator_cache.update_global_cache()
    model.train()
    model.reset_states()
    model.set_neighbor_sampler(args.train_sampler)
    model.memory.reset_memory()
    total, count = 0., 0
    for indexes in tqdm(loader, desc=f'Epoch {epoch} training', leave=False):
        idx = indexes.numpy()
        optimizer.zero_grad()
        out = model.compute_unsupervised_loss(
            src_node_ids=data.src_node_ids[idx], dst_node_ids=data.dst_node_ids[idx],
            node_interact_times=data.node_interact_times[idx], edge_ids=data.edge_ids[idx],
            epoch=epoch, save_logs_this_epoch=(epoch % 10 == 0),
            num_neighbors=args.num_neighbors, time_gap=args.time_gap)
        loss, payload = out if isinstance(out, (tuple, list)) and len(out) == 2 else (out, None)
        if not torch.isfinite(loss).all().item():
            raise ValueError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        if payload is not None:
            with torch.no_grad():
                src, dst, times, src_mem, dst_mem = payload
                model.memory.update_state(src, src_mem.detach(), times)
                model.memory.update_state(dst, dst_mem.detach(), times)
        total += loss.item()
        count += 1
    if not count:
        raise ValueError('Empty training split')
    return total / count


def run_selection(model, train_epoch, validate, epochs, output_dir):
    """Record both policies on the same states, with fixed stopping horizon."""
    if epochs < 1:
        raise ValueError('Fixed epoch count must be positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    best = {}
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = float(train_epoch(epoch))
        if not np.isfinite(train_loss):
            raise ValueError('Non-finite training loss')
        with isolated_validation(model, model.device):
            val_auroc = float(validate())
        if not np.isfinite(val_auroc) or not 0 <= val_auroc <= 1:
            raise ValueError('Invalid Validation AUROC')
        scores = {'train_loss': train_loss, 'val_auroc': val_auroc}
        history.append({'epoch': epoch, **scores})
        for policy, score in scores.items():
            previous = best.get(policy)
            improved = previous is None or (score < previous['score'] if policy == 'train_loss' else score > previous['score'])
            if improved:
                checkpoint = output_dir / f'{policy}_epoch{epoch:04d}.pkl'
                # Never overwrite an existing checkpoint, even inside a new run.
                with checkpoint.open('xb') as handle:
                    torch.save(model.state_dict(), handle)
                best[policy] = {'epoch': epoch, 'score': score, 'checkpoint': checkpoint.name,
                                'checkpoint_sha256': file_sha256(checkpoint),
                                'validation_labels_used_for_selection': policy == 'val_auroc'}
        logging.info('Epoch %d: train_loss=%.8f validation_auroc=%.8f', epoch, train_loss, val_auroc)
    record = {'fixed_epochs': epochs, 'policies': best, 'history': history}
    with (output_dir / 'selection.json').open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    return record


def main():
    if os.environ.get('AHEAD_SHARED_TRAINING_CONFIRM') != '1':
        raise ValueError('Real training needs prior authorization and AHEAD_SHARED_TRAINING_CONFIRM=1')
    args = get_node_classification_args()
    if args.model_name != 'AHEADGraphMixer' or min(args.num_runs, args.num_epochs, args.batch_size) < 1:
        raise ValueError('Invalid model, run count, fixed epoch count, or batch size')
    root_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root_dir:
        raise ValueError('Run from the src directory')
    mode = os.environ.get('AHEAD_DEGREE_CALIB_MODE', 'original')
    if mode not in ('original', 'none', 'continuous'):
        raise ValueError('Invalid degree calibration')
    run_id = os.environ.get('AHEAD_RESEARCH_RUN_ID', '')
    files = research_input_files(root_dir, args.dataset_name)
    files['train_shared_research.py'] = Path(__file__).resolve()
    files['evaluate_shared_research.py'] = root_dir / 'evaluate_shared_research.py'
    fingerprints = {key: {'sha256': file_sha256(path), 'bytes': path.stat().st_size} for key, path in files.items()}
    root = research_output_path(root_dir, run_id)
    root.mkdir(parents=True, exist_ok=False)
    provenance = {'schema_version': 1, 'protocol': 'shared_trajectory_fixed_epochs',
                  'dataset': args.dataset_name, 'calibration_mode': mode, 'configuration': vars(args).copy(),
                  'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
                  'pandas': pd.__version__, 'scikit_learn': sklearn.__version__, 'files': fingerprints}
    with (root / 'provenance.json').open('x') as handle:
        json.dump(provenance, handle, indent=2)
    nodes, edges, full, train, val, _test = get_node_classification_data(args.dataset_name, args.val_ratio, args.test_ratio)
    if np.any(np.diff(full.node_interact_times) < 0):
        raise ValueError('Interactions must be chronological')
    train_sampler = get_neighbor_sampler(train, args.sample_neighbor_strategy, args.time_scaling_factor, seed=0)
    val_sampler = get_neighbor_sampler(full, args.sample_neighbor_strategy, args.time_scaling_factor, seed=1)
    train_loader = get_idx_data_loader(list(range(len(train.src_node_ids))), args.batch_size, False)
    val_loader = get_idx_data_loader(list(range(len(val.src_node_ids))), args.batch_size, False)
    results = []
    for seed in range(args.num_runs):
        set_research_seed(seed, args.device)
        train_sampler.reset_random_state()
        val_sampler.reset_random_state()
        model = build_research_model(args, nodes, edges, train_sampler, seed).to(args.device)
        optimizer = create_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
        # Sampler is intentionally excluded from serialized CLI configuration.
        args.train_sampler = train_sampler
        def train_one(epoch):
            return run_train_epoch(model, optimizer, train_loader, train, args, epoch)
        def validate():
            val_sampler.reset_random_state()
            scores, labels = collect_research_scores(model, val_sampler, val_loader, val,
                                                     args.num_neighbors, args.time_gap, True)
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError('Validation requires both binary classes')
            return roc_auc_score(labels, scores)
        seed_dir = root / f'seed{seed}'
        record = run_selection(model, train_one, validate, args.num_epochs, seed_dir)
        del args.train_sampler
        results.append({'seed': seed, 'selection_sha256': file_sha256(seed_dir / 'selection.json'),
                        'policies': record['policies']})
    for name, path in files.items():
        if file_sha256(path) != fingerprints[name]['sha256']:
            raise RuntimeError(f'Input changed during training: {name}')
    for result in results:
        seed_dir = root / f"seed{result['seed']}"
        if file_sha256(seed_dir / 'selection.json') != result['selection_sha256']:
            raise RuntimeError('Selection record changed before sealing')
        for selected in result['policies'].values():
            if file_sha256(seed_dir / selected['checkpoint']) != selected['checkpoint_sha256']:
                raise RuntimeError('Selected checkpoint changed before sealing')
    seal = {'schema_version': 1, 'protocol': 'shared_trajectory_fixed_epochs',
            'provenance_sha256': file_sha256(root / 'provenance.json'), 'runs': results}
    with (root / 'selection_complete.json').open('x') as handle:
        json.dump(seal, handle, indent=2, allow_nan=False)
    logging.info('All shared-trajectory selections sealed; no Test evaluation performed.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
