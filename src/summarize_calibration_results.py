"""Summarize saved per-seed calibration metrics without inference or training.

Pairs are Fixed 1.1 minus each comparator. Input hashes are recorded for
traceability, not signatures; this program does not re-verify raw scores.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev

METHODS = ('original', 'none', 'continuous', 'fixed_1_1', 'c_pairwise', 'c_guarded')
BASES = tuple(x for x in METHODS if x != 'fixed_1_1')
COMPARISONS = {'fixed_1_1': BASES, 'c_pairwise': ('none','original','continuous'),
               'c_guarded': ('none','original','continuous')}
METRICS = ('AUROC', 'AP', 'PR_AUC')
POLICIES = ('val_auroc', 'train_loss')
ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{where}: metric must be a finite number in [0, 1]')
    return float(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_result(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError(f'Missing or symlinked result: {p}')
    before = digest(p)
    with p.open(encoding='utf-8') as f:
        result = json.load(f, object_pairs_hook=unique_object)
    if digest(p) != before:
        raise ValueError(f'Result changed while reading: {p}')
    return result, before


def unpack(report, source, source_hash):
    """Normalize authentic historical report structure or new per-run result."""
    if not isinstance(report, dict):
        raise ValueError('Result must be a JSON object')
    if 'datasets' in report:
        if not isinstance(report['datasets'], dict):
            raise ValueError('Historical datasets must be an object')
        if report.get('protocol') != 'fixed_degree_up_development_v1':
            raise ValueError('Unsupported historical report protocol')
        for dataset, obj in report['datasets'].items():
            context = {'format': 'historical_fixed_report', 'dataset': dataset,
                       'run_id': obj['run_id'], 'data_mode': 'HISTORICAL_UNSPECIFIED',
                       'experiment_mode': 'HISTORICAL_UNSPECIFIED', 'calibration_mode': 'historical',
                       'historical_report_sha256': source_hash,
                       'development_bounds': obj['development_bounds'],
                       'warning': 'Previously explored D; original scores lack event IDs; C-label release unknown.'}
            for policy, rows in obj['results'].items():
                for row in rows:
                    yield context, policy, row, source
        return
    required = ('schema_version', 'status', 'dataset', 'run_id', 'data_mode', 'experiment_mode',
                'data_manifest_sha256', 'selection_seal_sha256', 'development_seal_sha256',
                'source_code_sha256', 'algorithm_sha256', 'verifier_sha256', 'calibration_mode',
                'configuration', 'fixed_method_parameters', 'reference_bounds', 'development_bounds',
                'reference_events', 'reference_positives', 'development_events', 'development_positives',
                'label_scope', 'label_release_time', 'methods', 'policies', 'rows')
    missing = [key for key in required if key not in report]
    if missing:
        raise ValueError(f'Missing result fields: {missing}')
    if report['schema_version'] != 1 or report['status'] != 'COMPLETED_SCORES_ONLY':
        raise ValueError('Result is not a completed per-score evaluation')
    if report['calibration_mode'] not in ('historical', 'custom'):
        raise ValueError('Invalid calibration mode')
    if report['data_mode'] not in ('historical', 'custom', 'HISTORICAL_RUN_NOT_INTEGRATED'):
        raise ValueError('Invalid data mode')
    if report['experiment_mode'] not in ('historical', 'custom', 'HISTORICAL_RUN_NOT_INTEGRATED'):
        raise ValueError('Invalid experiment mode')
    for key in ('selection_seal_sha256', 'development_seal_sha256', 'source_code_sha256'):
        if not isinstance(report[key],str) or not re.fullmatch(r'[0-9a-f]{64}',report[key]):
            raise ValueError(f'Invalid {key}')
    if report['data_mode'] in ('historical', 'custom') and (not isinstance(report['data_manifest_sha256'],str)
            or not re.fullmatch(r'[0-9a-f]{64}',report['data_manifest_sha256'])):
        raise ValueError('Integrated result requires a manifest hash')
    if not isinstance(report['rows'], list) or not report['rows']:
        raise ValueError('No per-seed rows')
    context = {key: report[key] for key in required if key not in ('rows','methods','policies','status','schema_version')}
    context.update(format='calibration_result', warning='D explored; C methods use labels; label release may be unknown.')
    declared_methods, declared_policies = report['methods'], report['policies']
    if (not isinstance(declared_methods,list) or not declared_methods or len(set(declared_methods)) != len(declared_methods)
            or not set(declared_methods).issubset(METHODS) or not isinstance(declared_policies,list)
            or not declared_policies or len(set(declared_policies)) != len(declared_policies)
            or not set(declared_policies).issubset(POLICIES)):
        raise ValueError('Invalid declared methods or policies')
    seen = set()
    for row in report['rows']:
        key = (row['policy'], row['seed'])
        if key in seen:
            raise ValueError(f'Duplicate policy/seed row within {source}: {key}')
        seen.add(key)
        if row['policy'] not in declared_policies or set(row['metrics']) != set(declared_methods):
            raise ValueError('Rows do not match declared methods or policies')
        yield context, row['policy'], row, source


def normalize(inputs):
    groups = {}
    source_hashes = {}
    for path in inputs:
        report, source_hash = read_result(path)
        source = str(Path(path).resolve())
        if source in source_hashes:
            raise ValueError(f'Duplicate input file: {source}')
        source_hashes[source] = source_hash
        for context, policy, row, _ in unpack(report, source, source_hash):
            dataset, run_id = context.get('dataset'), context.get('run_id')
            if not isinstance(dataset,str) or not dataset or not isinstance(run_id,str) or not run_id:
                raise ValueError('Dataset and run_id must be non-empty strings')
            if policy not in POLICIES:
                raise ValueError(f'Invalid policy: {policy}')
            seed = row.get('seed')
            if isinstance(seed,bool) or not isinstance(seed,int) or seed < 0:
                raise ValueError('Seed must be a nonnegative integer')
            for key in ('score_sha256','checkpoint_sha256'):
                if not isinstance(row.get(key),str) or not re.fullmatch(r'[0-9a-f]{64}',row[key]):
                    raise ValueError(f'Missing or invalid {key}')
            group = groups.setdefault(dataset, {'context': context, 'rows': {}, 'sources': set()})
            if group['context'] != context:
                raise ValueError(f'{dataset}: incompatible run, data, modes, configuration, bounds or provenance')
            group['sources'].add(source)
            identity = (policy,seed)
            shared = {'score_sha256': row['score_sha256'], 'checkpoint_sha256': row['checkpoint_sha256']}
            if context['format'] == 'calibration_result':
                for key in ('score_metadata_sha256','selected_epoch','alignment'):
                    if key not in row: raise ValueError(f'Missing row provenance: {key}')
                    shared[key] = row[key]
                if (not isinstance(shared['score_metadata_sha256'],str)
                        or not re.fullmatch(r'[0-9a-f]{64}',shared['score_metadata_sha256'])
                        or isinstance(shared['selected_epoch'],bool) or not isinstance(shared['selected_epoch'],int)
                        or shared['selected_epoch'] < 1
                        or shared['alignment'] not in ('EVENT_IDS_MATCH','LEGACY_TIMESTAMPS_ONLY_EVENT_ID_UNVERIFIED')):
                    raise ValueError('Invalid row provenance')
            current = group['rows'].setdefault(identity, {'provenance':shared, 'metrics':{}})
            if current['provenance'] != shared:
                raise ValueError(f'{dataset}/{policy}/seed{seed}: incompatible score or checkpoint')
            metrics = row.get('metrics')
            if not isinstance(metrics,dict) or not metrics or not set(metrics).issubset(METHODS):
                raise ValueError('Invalid method metrics')
            for method, values in metrics.items():
                if method in current['metrics']:
                    raise ValueError(f'Duplicate result: {dataset}/{policy}/seed{seed}/{method}')
                if not isinstance(values,dict) or set(values) != set(METRICS):
                    raise ValueError(f'{dataset}/{policy}/seed{seed}/{method}: missing metric')
                current['metrics'][method] = {m:number(values[m],f'{dataset}/{policy}/{seed}/{method}/{m}') for m in METRICS}
    return groups, source_hashes


def stats(values):
    return {'n': len(values), 'mean': mean(values) if values else None,
            'sample_sd': stdev(values) if len(values) > 1 else None}


def summarize(groups, datasets=None, policies=None, methods=None, seeds=None):
    selected_ds = sorted(groups) if datasets is None else list(datasets)
    selected_policies = POLICIES if policies is None else tuple(policies)
    selected_methods = METHODS if methods is None else tuple(methods)
    if not selected_ds or len(set(selected_ds)) != len(selected_ds) or not set(selected_ds).issubset(groups):
        raise ValueError('Requested dataset missing or duplicate')
    if not selected_policies or len(set(selected_policies)) != len(selected_policies) or not set(selected_policies).issubset(POLICIES):
        raise ValueError('Invalid or duplicate policies')
    if not selected_methods or len(set(selected_methods)) != len(selected_methods) or not set(selected_methods).issubset(METHODS):
        raise ValueError('Invalid or duplicate methods')
    if seeds is not None and (not seeds or len(set(seeds)) != len(seeds) or any(isinstance(x,bool) or not isinstance(x,int) or x < 0 for x in seeds)):
        raise ValueError('Invalid or duplicate seeds')
    summaries, paired, contexts = [], [], {}
    for dataset in selected_ds:
        group = groups[dataset]
        contexts[dataset] = {'experiment': group['context'], 'sources': sorted(group['sources'])}
        for policy in selected_policies:
            records = {seed:value for (pol,seed),value in group['rows'].items() if pol == policy and (seeds is None or seed in seeds)}
            for method in selected_methods:
                for metric in METRICS:
                    values = {seed:r['metrics'][method][metric] for seed,r in records.items() if method in r['metrics']}
                    summaries.append({'dataset':dataset,'policy':policy,'method':method,'metric':metric,
                                      'seeds':sorted(values),**stats(list(values.values())),
                                      'status':('METHOD_OR_POLICY_MISSING' if not values else
                                                'MISSING_REQUESTED_SEEDS' if seeds is not None and set(values) != set(seeds) else
                                                'PARTIAL_SEEDS' if set(values) != set(records) else 'OK')})
            for candidate, bases in COMPARISONS.items():
                if candidate not in selected_methods:
                    continue
                for base in bases:
                    for metric in METRICS:
                        chosen = {s:r for s,r in records.items() if candidate in r['metrics']}
                        ref = {s:r for s,r in records.items() if base in r['metrics']} if base in selected_methods else {}
                        if not chosen or not ref:
                            item = {'status':'BASE_NOT_SELECTED' if base not in selected_methods else 'METHOD_OR_POLICY_MISSING',
                                    'seeds':[], **stats([])}
                        elif set(chosen) != set(ref) or (seeds is not None and set(chosen) != set(seeds)):
                            item = {'status':'UNMATCHED_SEEDS', 'candidate_seeds':sorted(chosen), 'base_seeds':sorted(ref),
                                    'seeds':[], **stats([])}
                        else:
                            paired_seeds = sorted(chosen)
                            differences = [chosen[s]['metrics'][candidate][metric] - ref[s]['metrics'][base][metric] for s in paired_seeds]
                            item = {'status':'OK','seeds':paired_seeds, **stats(differences)}
                        paired.append({'dataset':dataset,'policy':policy,'candidate':candidate,
                                       'baseline':base,'metric':metric,**item})
    return {'schema_version':1, 'comparison':'candidate minus baseline, paired by dataset/seed/policy/checkpoint/scores',
            'sample_sd':'ddof=1; null for fewer than two seeds', 'contexts':contexts,
            'aggregates':summaries, 'paired_differences':paired,
            'verification_scope':'Saved JSON consistency only; raw scores, data and checkpoint hashes are not recomputed.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, type=Path)
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--policies', nargs='+', choices=POLICIES)
    parser.add_argument('--methods', nargs='+', choices=METHODS)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        destination = args.output.absolute()
        if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
            raise ValueError('Output already exists or parent directory is missing')
        if destination.resolve().is_relative_to(ROOT):
            raise ValueError('Output must be outside the original src directory')
        groups, sources = normalize(args.input)
        result = summarize(groups,args.datasets,args.policies,args.methods,args.seeds)
        result['source_sha256'] = sources
        if any(digest(path) != sha for path,sha in sources.items()):
            raise ValueError('An input result changed during aggregation')
        for item in result['aggregates']:
            sd = 'N/A' if item['sample_sd'] is None else f"{item['sample_sd']:.9f}"
            avg = 'N/A' if item['mean'] is None else f"{item['mean']:.9f}"
            print(f"{item['dataset']} {item['policy']} {item['method']} {item['metric']} n={item['n']} mean={avg} sample_sd={sd} {item['status']}")
        for item in result['paired_differences']:
            sd = 'N/A' if item['sample_sd'] is None else f"{item['sample_sd']:.9f}"
            avg = 'N/A' if item['mean'] is None else f"{item['mean']:+.9f}"
            print(f"paired {item['dataset']} {item['policy']} {item['candidate']} - {item['baseline']} {item['metric']} n={item['n']} mean={avg} sample_sd={sd} {item['status']}")
        with destination.open('x',encoding='utf-8') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print(f'Saved {destination}; saved JSON statistics only, no inference.')
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'Statistics failed: {exc}',file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
