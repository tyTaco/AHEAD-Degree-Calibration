"""Generate sealed C/D scores from a completed isolated research run.

Reuses the historical development scorer and split; never modifies the original
research project. Inference runs only after explicit --confirm-scores.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from degree_development import batch_completion_times, chronological_development_split, paired_development_scores
from evaluate_preprocessed_calibration import (ALGORITHM_FILES, VERIFIER_FILES, METHODS, POLICIES, evaluate, verified_integration)
from run_degree_development import (_data, _pad_features, collect_scores_and_degrees)
from run_label_gate_development import _verify_run
from train_research import build_research_model, file_sha256, set_research_seed
from utils.DataLoader import get_idx_data_loader
from utils.utils import get_neighbor_sampler

ROOT = Path(__file__).resolve().parent


def collect_with_ids(model, sampler, data, batch_size, num_neighbors, time_gap):
    """Capture IDs passed to the existing scorer as each actual batch is yielded."""
    captured = []
    class RecordingLoader:
        def __iter__(self):
            for indexes in get_idx_data_loader(list(range(len(data.edge_ids))), batch_size, False):
                positions = indexes.numpy()
                captured.append(np.asarray(data.edge_ids[positions]).copy())
                yield indexes
    scores, degrees = collect_scores_and_degrees(model, sampler, RecordingLoader(), data, num_neighbors, time_gap)
    if not captured:
        raise ValueError('No scored events')
    ids = np.concatenate(captured)
    if len(ids) != len(scores) or len(ids) != len(degrees) or not np.array_equal(ids, data.edge_ids):
        raise ValueError('Scored event IDs or event order do not match the requested segment')
    return scores, degrees, ids


def add_scoring_source(project):
    """Register this bridge without replacing any workspace file."""
    source, target = Path(__file__).resolve(), project / Path(__file__).name
    sha = file_sha256(source)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or file_sha256(target) != sha:
            raise ValueError('Workspace scoring bridge already exists with a different hash')
    else:
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if file_sha256(target) != sha:
            raise ValueError('Scoring bridge copy mismatch')
    return sha


def score(project, run_id, seeds, policies, *, source=None, confirmed=False):
    if not confirmed:
        raise ValueError('Explicit --confirm-scores is required for CPU inference')
    project = Path(project).resolve(strict=True)
    if project == ROOT or project.is_relative_to(ROOT):
        raise ValueError('Refusing to generate scores in the original research src')
    integrated = verified_integration(project, run_id, True, source)
    for name in set(ALGORITHM_FILES + VERIFIER_FILES):
        target = project / name
        if target.is_symlink() or file_sha256(target) != file_sha256(ROOT / name):
            raise ValueError(f'Calibration or verifier code differs from workspace: {name}')
    root, provenance, runs = _verify_run(project, run_id)
    if integrated['dataset'] != provenance['dataset'] or integrated['configuration'] != provenance['configuration']:
        raise ValueError('Training provenance and integration record differ')
    if integrated['calibration'] != provenance['calibration_mode']:
        raise ValueError('Training calibration differs from integration record')
    config = SimpleNamespace(**provenance['configuration'])
    if (config.val_ratio != .05 or config.test_ratio != .5 or config.num_neighbors != 20
            or config.sample_neighbor_strategy != 'recent'
            or provenance['calibration_mode'] not in ('original','none','continuous')):
        raise ValueError('Run does not match the existing C/D scoring protocol')
    if integrated.get('experiment_mode') == 'historical' and provenance['calibration_mode'] != 'original':
        raise ValueError('Historical training must use original calibration')
    indexed = {r['seed']: r for r in runs}
    if (not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(indexed)
            or not policies or len(set(policies)) != len(policies) or not set(policies).issubset(POLICIES)):
        raise ValueError('Invalid seed or checkpoint policy')
    destination = root / 'degree_development'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Scoring output already exists; refusing to overwrite')
    dataset = provenance['dataset']
    csv = project / 'processed_data' / dataset / f'ml_{dataset}.csv'
    metadata = pd.read_csv(csv, usecols=['u', 'i', 'ts', 'idx'])
    times = metadata.ts.to_numpy(dtype=np.float64)
    parts = chronological_development_split(times)
    if parts['selection'].stop != np.searchsorted(times, np.quantile(times, 1-config.test_ratio), side='right'):
        raise ValueError('Selection boundary differs from training')
    d = parts['development']
    labels = pd.read_csv(csv, usecols=['label'], skiprows=range(1, d.start+1), nrows=d.stop-d.start)['label'].to_numpy()
    if len(labels) != d.stop-d.start or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Development labels must contain both classes')
    nodes = _pad_features(project / 'processed_data' / dataset / f'ml_{dataset}_node.npy')
    edges = _pad_features(project / 'processed_data' / dataset / f'ml_{dataset}.npy')
    prefix = _data(metadata.iloc[:d.stop])
    selection, reference, development = (_data(metadata.iloc[parts[k]]) for k in ('selection','reference','development'))
    model_ready = selection.node_interact_times[-1].item()
    ref_time, dev_time = reference.node_interact_times, development.node_interact_times
    if not model_ready < ref_time.min() <= ref_time.max() < dev_time.min():
        raise ValueError('C/D time boundaries overlap')
    ref_available = batch_completion_times(ref_time, config.batch_size)
    dev_available = batch_completion_times(dev_time, config.batch_size)
    add_scoring_source(project)
    names = ('score_preprocessed_research.py', 'run_degree_development.py', 'degree_development.py', 'degree_ecdf_prototype.py')
    code = {name:file_sha256(project/name) for name in names}
    for name in names:
        if code[name] != file_sha256(ROOT/name):
            raise ValueError(f'Scoring implementation differs from workspace: {name}')
    destination.mkdir(exist_ok=False)
    protocol = {'schema_version':1, 'run_id':run_id, 'dataset':dataset,
        'input_provenance_sha256':file_sha256(root/'provenance.json'),
        'selection_seal_sha256':file_sha256(root/'selection_complete.json'), 'development_code_sha256':code,
        'segments':{name:[part.start,part.stop] for name,part in parts.items()},
        'model_ready_time':model_ready, 'degree_cap':20, 'prior_strength':50.0, 'batch_size':config.batch_size,
        'selection_policies':list(policies), 'requested_seeds':list(seeds),
        'scoring_protocol':'offline_batch_delayed_within_batch_future_candidates',
        'event_id_capture':'batch_indices_used_for_compute_anomaly_score',
        'training_calibration':provenance['calibration_mode'], 'scoring_calibration':'none',
        'score_time_semantics':'last_event_timestamp_in_each_batch', 'development_labels_used_for_metrics':True}
    with (destination/'development_started.json').open('x') as out:
        json.dump(protocol,out,indent=2,allow_nan=False)
    old_mode = os.environ.get('AHEAD_DEGREE_CALIB_MODE')
    os.environ['AHEAD_DEGREE_CALIB_MODE']='none'
    artifacts = {}
    try:
        config.device='cpu'
        for seed in seeds:
            for policy in policies:
                set_research_seed(seed,'cpu')
                sampler=get_neighbor_sampler(prefix,config.sample_neighbor_strategy,config.time_scaling_factor,seed=1)
                model=build_research_model(config,nodes,edges,sampler,seed).to('cpu')
                chosen=indexed[seed]['policies'][policy]
                checkpoint=root/f'seed{seed}'/chosen['checkpoint']
                if checkpoint.is_symlink() or file_sha256(checkpoint)!=chosen['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed before scoring')
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
                model.memory_generator.grad_mask_dict=None
                # Preserve the existing B replay and sequential C then D state.
                collect_with_ids(model,sampler,selection,config.batch_size,config.num_neighbors,config.time_gap)
                ref_scores,ref_degrees,ref_ids=collect_with_ids(model,sampler,reference,config.batch_size,config.num_neighbors,config.time_gap)
                raw,degrees,dev_ids=collect_with_ids(model,sampler,development,config.batch_size,config.num_neighbors,config.time_gap)
                values,metrics=paired_development_scores(ref_scores,ref_degrees,ref_available,raw,degrees,dev_available,
                    labels,model_ready_time=model_ready,prior_strength=50.)
                item={'seed':seed,'policy':policy,'selected_epoch':chosen['epoch'],
                      'checkpoint_sha256':chosen['checkpoint_sha256'],
                      'validation_labels_used_for_selection':policy=='val_auroc','metrics':metrics}
                metric_path=destination/f'{policy}_seed{seed}_metrics.json'
                scores_path=destination/f'{policy}_seed{seed}_scores.npz'
                with metric_path.open('x') as out: json.dump(item,out,indent=2,allow_nan=False)
                with scores_path.open('xb') as out:
                    np.savez_compressed(out,reference_scores=ref_scores,reference_degrees=ref_degrees,
                        reference_times=ref_time,reference_event_ids=ref_ids,reference_score_available_at=ref_available,
                        development_degrees=degrees,development_times=dev_time,development_event_ids=dev_ids,
                        development_score_available_at=dev_available,**values)
                for p in (metric_path,scores_path): artifacts[p.name]=file_sha256(p)
        _verify_run(project,run_id)
        verified_integration(project,run_id,True,source)
        if any(file_sha256(project/name)!=digest or file_sha256(ROOT/name)!=digest for name,digest in code.items()):
            raise ValueError('Scoring code changed during inference')
        if any(file_sha256(destination/name)!=digest for name,digest in artifacts.items()):
            raise ValueError('Scoring artifact changed before sealing')
        with (destination/'development_complete.json').open('x') as out:
            json.dump({'num_runs':len(runs),'scored_seeds':list(seeds),'scored_policies':list(policies),
                       'protocol_sha256':file_sha256(destination/'development_started.json'),
                       'artifact_sha256':artifacts},out,indent=2,allow_nan=False)
        return file_sha256(destination/'development_complete.json')
    finally:
        if old_mode is None: os.environ.pop('AHEAD_DEGREE_CALIB_MODE',None)
        else: os.environ['AHEAD_DEGREE_CALIB_MODE']=old_mode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dataset',required=True,choices=('wikipedia','mooc','bitcoinalpha'))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,required=True)
    parser.add_argument('--policies',nargs='+',choices=POLICIES,required=True)
    parser.add_argument('--methods',nargs='+',choices=METHODS+('all',),required=True)
    parser.add_argument('--mode',choices=('historical','custom'),default='historical')
    parser.add_argument('--fixed-multiplier',type=float)
    parser.add_argument('--block-size',type=int)
    parser.add_argument('--replicates',type=int)
    parser.add_argument('--bootstrap-seed',type=int)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirm-scores',action='store_true')
    a=parser.parse_args(argv)
    try:
        if not a.confirm_scores: raise ValueError('Explicit --confirm-scores is required')
        project=a.project_root.resolve(strict=True)
        info=verified_integration(project,a.run_id,True,a.source)
        if info['dataset'] != a.dataset: raise ValueError('Dataset mismatch')
        if a.mode=='historical' and any(x is not None for x in (a.fixed_multiplier,a.block_size,a.replicates,a.bootstrap_seed)):
            raise ValueError('Historical calibration parameters are fixed')
        opts={'fixed_multiplier':a.fixed_multiplier if a.fixed_multiplier is not None else 1.1,
              'block_size':a.block_size if a.block_size is not None else 200,
              'replicates':a.replicates if a.replicates is not None else 300,
              'bootstrap_seed':a.bootstrap_seed if a.bootstrap_seed is not None else 20261121}
        if (not np.isfinite(opts['fixed_multiplier']) or opts['fixed_multiplier'] <= 0 or opts['block_size'] < 1
                or opts['replicates'] < 20 or opts['bootstrap_seed'] < 0): raise ValueError('Invalid calibration settings')
        output=a.output.absolute()
        if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.resolve().is_relative_to(ROOT):
            raise ValueError('Output must be a new file outside original src')
        sha=score(project,a.run_id,a.seeds,a.policies,source=a.source,confirmed=True)
        methods=METHODS if 'all' in a.methods else tuple(dict.fromkeys(a.methods))
        result=evaluate(project,a.run_id,a.dataset,a.seeds,a.policies,methods,a.mode,opts,sha,
                        require_integration=True,source_override=a.source)
        with output.open('x') as out: json.dump(result,out,indent=2,allow_nan=False)
        print(json.dumps({'status':result['status'],'development_seal_sha256':sha,'output':str(output)}))
        return 0
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,json.JSONDecodeError) as exc:
        print(f'Score integration failed: {exc}',file=sys.stderr);return 1

if __name__=='__main__': sys.exit(main())
