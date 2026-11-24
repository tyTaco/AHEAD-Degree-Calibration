"""Label-informed retrospective development on already explored scores.

This is not AHEAD's method or an independent Test evaluator. Earlier labels'
actual release times may be unknown or depend on later events. The caller must
verify input provenance, exclude Test labels, and report those limitations.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from label_gate_development import _metrics


def chronological_cut(event_times, available_times, *, batch_size):
    """Choose the closest valid batch boundary to the midpoint without labels."""
    times, available = np.asarray(event_times), np.asarray(available_times)
    if (times.ndim != 1 or available.ndim != 1 or not len(times) or len(times) != len(available)
            or any(x.dtype.kind not in 'iuf' or not np.isfinite(x).all() for x in (times, available))
            or any(x.dtype.kind == 'f' and x.dtype.itemsize < 8 for x in (times, available))
            or np.any(times >= 2**53) or np.any(times <= -(2**53))
            or np.any(available >= 2**53) or np.any(available <= -(2**53))
            or np.any(times[1:] < times[:-1])
            or isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError('Expected aligned, ordered precise timestamps and positive batch size')
    for start in range(0, len(times), batch_size):
        stop = min(start + batch_size, len(times))
        if np.any(available[start:stop] != times[stop - 1]):
            raise ValueError('Score availability must equal batch completion time')
    boundaries = [i for i in range(batch_size, len(times), batch_size) if times[i - 1] < times[i]]
    if not boundaries:
        raise ValueError('No strictly separated batch boundary exists')
    cut = min(boundaries, key=lambda i: (abs(i - len(times) / 2), i))
    if np.any(available[:cut] >= times[cut]):
        raise ValueError('Selection scores must be available before evaluation events')
    return cut


def _inputs(scores, degrees):
    raw, degree = np.asarray(scores), np.asarray(degrees)
    if (raw.ndim != 1 or degree.ndim != 1 or not len(raw) or len(raw) != len(degree)
            or raw.dtype.kind not in 'iuf' or degree.dtype.kind not in 'iuf'
            or not np.isfinite(raw).all() or not np.isfinite(degree).all()
            or np.any(raw < 0) or np.any(degree < 0) or np.any(degree > 20)
            or np.any(degree != np.floor(degree))):
        raise ValueError('Expected aligned nonnegative scores and sampled degrees from 0 to 20')
    if raw.dtype.kind in 'iu' and np.any(raw > 2**53):
        raise ValueError('Integer scores exceed exact float64 range')
    raw64 = raw.astype(np.float64)
    if (not np.isfinite(raw64).all() or (raw.dtype.kind == 'f'
            and not np.array_equal(raw64.astype(raw.dtype), raw))):
        raise ValueError('Scores must be exactly representable as float64')
    return raw64, degree


def choose_factor(scores, degrees, historical_labels):
    """Choose 1, 0.9 or 1.1 on earlier labels; ties prefer no adjustment.

    A valid one-class historical segment explicitly falls back to 1. Invalid,
    missing, or misaligned labels are rejected; no exception is hidden. This
    does not establish when labels became known, or imply future improvement.
    """
    raw, degree = _inputs(scores, degrees)
    labels = np.asarray(historical_labels)
    if labels.ndim != 1 or len(labels) != len(raw) or labels.dtype.kind not in 'biuf':
        raise ValueError('Expected aligned binary historical labels')
    classes = set(np.unique(labels))
    if not classes or not classes.issubset({0, 1}):
        raise ValueError('Expected aligned binary historical labels')
    if len(classes) == 1:
        return {'factor': 1., 'reason': 'single_class_fallback', 'historical_auroc': {}}
    best_factor, best_auc, metrics = 1., -np.inf, {}
    for factor in (1., .9, 1.1):
        adjusted = raw * np.where(degree < 3, factor, 1.)
        if not np.isfinite(adjusted).all():
            raise ValueError('Adjusted scores overflow')
        value = float(roc_auc_score(labels, adjusted))
        metrics[str(factor)] = value
        if value > best_auc:
            best_factor, best_auc = factor, value
    return {'factor': best_factor, 'reason': 'historical_auroc', 'historical_auroc': metrics}


def fixed_scores(scores, degrees, selection):
    """Freeze four score arrays without accepting evaluation labels."""
    raw, degree = _inputs(scores, degrees)
    if (not isinstance(selection, dict) or selection.get('factor') not in (1., .9, 1.1)
            or selection.get('reason') not in ('historical_auroc', 'single_class_fallback')
            or (selection.get('reason') == 'single_class_fallback' and selection.get('factor') != 1.)):
        raise ValueError('Expected a valid historical selection')
    with np.errstate(over='ignore', invalid='ignore'):
        values = {'none': raw.copy(),
                  'original': raw * np.where(degree < 3, .9, 1.),
                  'continuous': raw * (1. - .1 * np.exp(-degree / 3.)),
                  'pairwise': raw * np.where(degree < 3, selection['factor'], 1.)}
    if any(not np.isfinite(x).all() for x in values.values()):
        raise ValueError('Adjusted scores overflow')
    return values


def evaluate_frozen(scores_by_method, evaluation_labels):
    """Report metrics after score selection; no method choice is performed."""
    if not isinstance(scores_by_method, dict) or set(scores_by_method) != {'none', 'original', 'continuous', 'pairwise'}:
        raise ValueError('Expected all four fixed score arrays')
    labels = np.asarray(evaluation_labels)
    if labels.ndim != 1 or not len(labels) or labels.dtype.kind not in 'biuf' or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Evaluation labels must contain both binary classes')
    result = {}
    for name, scores in scores_by_method.items():
        scores = np.asarray(scores)
        if (scores.ndim != 1 or len(scores) != len(labels) or scores.dtype.kind not in 'iuf'
                or not np.isfinite(scores).all()):
            raise ValueError('Scores must be finite and aligned with evaluation labels')
        result[name] = _metrics(labels, scores)
    return result
