"""Exploratory label-supervised guard for retrospective degree calibration.

Both candidates are compared to None with paired time-block bootstrap. The
2.5% cutoff is a heuristic for two comparisons, not a confidence or safety
guarantee under temporal dependence, distribution shift, or selection bias.
Never use Test labels; label availability must be established independently.
"""
import numpy as np

from calibration_selection_prototype import select_with_block_bootstrap
from retrospective_pairwise_development import _inputs, choose_factor


def choose_guarded_factor(scores, degrees, historical_labels, *, block_size=200,
                          replicates=300, seed=20261121):
    """Choose 0.9/1.1 only with positive paired bootstrap lower percentile.

    The historical segment is supervised. Ties and explicit insufficiency
    fall back to 1.0; malformed inputs and unrelated exceptions are raised.
    Future labels and scores are not accepted by this function.
    """
    for name, value, minimum in (('block_size', block_size, 1), ('replicates', replicates, 20), ('seed', seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    raw, degree = _inputs(scores, degrees)
    preliminary = choose_factor(raw, degree, historical_labels)
    if preliminary['reason'] == 'single_class_fallback':
        return {'factor': 1., 'reason': 'single_class_fallback', 'candidates': {}}
    if len(raw) <= block_size:
        return {'factor': 1., 'reason': 'insufficient_blocks', 'candidates': {}}
    y = np.asarray(historical_labels)
    baseline_auc = preliminary['historical_auroc']['1.0']
    qualified, evidence = [], {}
    for factor in (.9, 1.1):
        observed = preliminary['historical_auroc'][str(factor)] - baseline_auc
        if observed <= 0:
            evidence[str(factor)] = {'observed_delta': observed, 'bootstrap_lower': None,
                                     'valid_replicates': 0, 'qualified': False}
            continue
        candidate = raw * np.where(degree < 3, factor, 1.)
        try:
            result = select_with_block_bootstrap(y, raw, candidate, block_size=block_size,
                replicates=replicates, lower_quantile=.025, seed=seed)
        except ValueError as exc:
            if str(exc) != 'Too few bootstrap samples contain both label classes':
                raise
            return {'factor': 1., 'reason': 'insufficient_valid_bootstraps', 'candidates': evidence}
        evidence[str(factor)] = {'observed_delta': result.observed_delta,
            'bootstrap_lower': result.bootstrap_lower, 'valid_replicates': result.valid_replicates,
            'qualified': result.use_candidate}
        if result.use_candidate:
            qualified.append((result.observed_delta, factor))
    if not qualified:
        return {'factor': 1., 'reason': 'no_candidate_passed', 'candidates': evidence}
    # If equal, prefer 0.9 through the fixed candidate order.
    winner = max(qualified, key=lambda item: item[0])[1]
    return {'factor': winner, 'reason': 'bootstrap_guard_passed', 'candidates': evidence}
