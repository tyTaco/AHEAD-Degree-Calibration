"""One-shot, CPU-only, label-informed gate analysis of saved development scores.

Never pass Test labels. This reads D labels, so the whole D interval becomes
exploratory data if its results are used to choose or revise a method.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_shared_research import load_verified_shared
from label_gate_development import evaluate_gate
from train_research import file_sha256, research_output_path
from verify_segmented_research import verify as verify_segmented


def _verify_run(project, run_id):
    root = research_output_path(project, run_id)
    provenance = root / 'provenance.json'
    if provenance.is_symlink():
        raise ValueError('Redirected provenance')
    protocol = json.loads(provenance.read_text()).get('protocol')
    if protocol == 'shared_trajectory_fixed_epochs':
        return load_verified_shared(project, run_id)
    if protocol == 'segmented_shared_trajectory_v1':
        return verify_segmented(project, run_id)
    raise ValueError('Unsupported training protocol')


def run(project_dir, run_id, confirmed):
    if not confirmed:
        raise ValueError('Explicit --confirm-development is required')
    project = Path(project_dir).resolve()
    if Path.cwd().resolve() != project:
        raise ValueError('Run from the src directory')
    root, provenance, runs = _verify_run(project, run_id)
    dev = root / 'degree_development'
    destination = root / 'label_gate_development'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Gate analysis already exists; refusing overwrite')
    if dev.is_symlink() or not dev.is_dir():
        raise ValueError('Missing or redirected development directory')
    protocol_path, seal_path = dev / 'development_started.json', dev / 'development_complete.json'
    if protocol_path.is_symlink() or seal_path.is_symlink():
        raise ValueError('Redirected development seals')
    development_seal_hash = file_sha256(seal_path)
    protocol, seal = json.loads(protocol_path.read_text()), json.loads(seal_path.read_text())
    if (file_sha256(protocol_path) != seal.get('protocol_sha256')
            or protocol.get('run_id') != run_id or protocol.get('dataset') != provenance['dataset']
            or protocol.get('input_provenance_sha256') != file_sha256(root / 'provenance.json')
            or protocol.get('selection_seal_sha256') != file_sha256(root / 'selection_complete.json')
            or seal.get('num_runs') != len(runs)):
        raise ValueError('Development provenance or seal mismatch')
    lo, hi = protocol['segments']['development']
    test_lo, _ = protocol['segments']['old_test']
    if not isinstance(lo, int) or not isinstance(hi, int) or not 0 <= lo < hi <= test_lo:
        raise ValueError('Development interval overlaps old Test')
    labels = pd.read_csv(project / 'processed_data' / provenance['dataset'] / f"ml_{provenance['dataset']}.csv",
                         usecols=['label'], skiprows=range(1, lo + 1), nrows=hi - lo)['label'].to_numpy()
    if len(labels) != hi - lo or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Development labels must contain both classes')
    code_paths = [project / name for name in ('run_label_gate_development.py', 'label_gate_development.py',
                                               'calibration_selection_prototype.py')]
    code_hashes = {path.name: file_sha256(path) for path in code_paths}
    artifacts = seal.get('artifact_sha256', {})
    if (not isinstance(artifacts, dict) or any(Path(name).name != name or (dev / name).is_symlink()
                                               for name in artifacts)):
        raise ValueError('Invalid or redirected development artifacts')
    results = {}
    for policy in ('val_auroc', 'train_loss'):
        results[policy] = []
        for item in runs:
            seed = item['seed']
            name = f'{policy}_seed{seed}_scores.npz'
            path = dev / name
            if path.is_symlink() or name not in artifacts or file_sha256(path) != artifacts[name]:
                raise ValueError('Development score artifact changed')
            with np.load(path, allow_pickle=False) as values:
                result = evaluate_gate(labels, values['none'], values['original'], values['development_times'],
                                       values['development_score_available_at'], batch_size=protocol['batch_size'])
            result['seed'] = seed
            result['checkpoint_sha256'] = item['policies'][policy]['checkpoint_sha256']
            result['score_sha256'] = artifacts[name]
            results[policy].append(result)
    summary = {}
    for policy, items in results.items():
        summary[policy] = {'original_chosen': sum(x['selection']['method'] == 'original' for x in items),
                           'paired_delta_vs_original': {}, 'paired_delta_vs_none': {}}
        for metric in ('AUROC', 'AP', 'PR_AUC'):
            gate = np.array([x['metrics']['gate'][metric] for x in items])
            for comparator in ('original', 'none'):
                delta = gate - np.array([x['metrics'][comparator][metric] for x in items])
                summary[policy][f'paired_delta_vs_{comparator}'][metric] = {
                    'mean': float(delta.mean()),
                    'sample_sd': float(delta.std(ddof=1)) if len(delta) > 1 else None}
    _verify_run(project, run_id)
    if (file_sha256(protocol_path) != seal['protocol_sha256'] or file_sha256(seal_path) != development_seal_hash
            or any(file_sha256(dev / name) != digest for name, digest in artifacts.items())
            or any(file_sha256(path) != code_hashes[path.name] for path in code_paths)):
        raise ValueError('Inputs changed during gate evaluation')
    report = {'schema_version': 1, 'run_id': run_id, 'dataset': provenance['dataset'],
              'label_scope': 'development_only; selection uses earlier labels; not an independent test',
              'bootstrap': {'block_size': protocol['batch_size'], 'replicates': 300,
                            'lower_quantile': .05, 'seed': 20261121},
              'development_seal_sha256': development_seal_hash,
              'code_sha256': code_hashes, 'results': results, 'summary': summary}
    destination.mkdir(exist_ok=False)
    with (destination / 'gate_complete.json').open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--confirm-development', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(Path(__file__).resolve().parent, args.run_id, args.confirm_development)['summary'], indent=2))


if __name__ == '__main__':
    main()
