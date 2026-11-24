"""Reproducible exploratory label-informed calibration gate on saved scores.

This is not an AHEAD paper method, not unsupervised, and not a Test evaluator.
Input labels must be from development only. The caller must verify artifact
provenance and keep previously explored data separate from independent tests.
"""
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from calibration_selection_prototype import select_with_block_bootstrap


def _metrics(y, scores):
    precision, recall, _ = precision_recall_curve(y, scores)
    return {'AUROC': float(roc_auc_score(y, scores)),
            'AP': float(average_precision_score(y, scores)),
            'PR_AUC': float(auc(recall, precision))}


def evaluate_gate(labels, none_scores, original_scores, event_times, available_times, *,
                  batch_size=200, replicates=300, lower_quantile=.05, seed=20261121):
    """Compare fixed Original/None and a past-label-selected gate on later data.

    The closest batch boundary to the midpoint with strictly increasing times
    is fixed without labels. The earlier segment selects; the later evaluates.
    A valid batch timestamp is the last event time of that batch. If either
    segment lacks both classes or bootstrap has too few valid samples, fail
    closed rather than silently selecting a method.
    """
    y, raw, original, times, available = map(np.asarray,
        (labels, none_scores, original_scores, event_times, available_times))
    arrays = (y, raw, original, times, available)
    if (any(a.ndim != 1 or len(a) == 0 for a in arrays)
            or len({len(a) for a in arrays}) != 1
            or set(np.unique(y)) != {0, 1}
            or any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all()
                   for a in (raw, original, times, available))
            or (times.dtype.kind == 'f' and times.dtype.itemsize < 8)
            or (available.dtype.kind == 'f' and available.dtype.itemsize < 8)
            or np.any(times >= 2**53) or np.any(times <= -(2**53))
            or np.any(available >= 2**53) or np.any(available <= -(2**53))
            or np.any(times[1:] < times[:-1])
            or isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1):
        raise ValueError('Expected aligned binary labels, finite scores and precise ordered times')
    for start in range(0, len(times), batch_size):
        stop = min(start + batch_size, len(times))
        if np.any(available[start:stop] != times[stop - 1]):
            raise ValueError('Score availability must equal batch completion time')
    boundaries = [i for i in range(batch_size, len(y), batch_size) if times[i - 1] < times[i]]
    if not boundaries:
        raise ValueError('No strictly separated batch boundary exists')
    cut = min(boundaries, key=lambda i: (abs(i - len(y) / 2), i))
    if np.any(available[:cut] >= times[cut]):
        raise ValueError('Selection scores must be available before evaluation events')
    if set(np.unique(y[:cut])) != {0, 1} or set(np.unique(y[cut:])) != {0, 1}:
        raise ValueError('Each segment must contain both classes')
    selection = select_with_block_bootstrap(y[:cut], raw[:cut], original[:cut],
        block_size=batch_size, replicates=replicates, lower_quantile=lower_quantile, seed=seed)
    selected = original[cut:] if selection.use_candidate else raw[cut:]
    return {'selection': {'method': 'original' if selection.use_candidate else 'none',
                          'delta_auroc': selection.observed_delta,
                          'bootstrap_lower': selection.bootstrap_lower,
                          'valid_replicates': selection.valid_replicates,
                          'requested_replicates': selection.requested_replicates},
            'split': {'index': cut, 'selection_count': cut, 'evaluation_count': len(y) - cut,
                      'selection_positive_count': int(y[:cut].sum()),
                      'evaluation_positive_count': int(y[cut:].sum()),
                      'last_selection_available_at': float(available[cut - 1]),
                      'first_evaluation_time': float(times[cut])},
            'metrics': {'none': _metrics(y[cut:], raw[cut:]),
                        'original': _metrics(y[cut:], original[cut:]),
                        'gate': _metrics(y[cut:], selected)},
            'label_scope': 'development; selection uses earlier labels; not an independent test'}
