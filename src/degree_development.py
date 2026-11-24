"""Predeclared temporal splits and paired development-score comparison.

This is a research utility, not a training or Test-evaluation entry point.
The caller must verify score/checkpoint provenance and the label policy before
calling it. In particular, arrays passed as development labels must never be
drawn from the previously used Test split.
"""
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from degree_ecdf_prototype import DegreeConditionalECDF


def batch_completion_times(timestamps, batch_size):
    """Return when offline within-batch scores become available, not event time."""
    times = np.asarray(timestamps)
    if (times.ndim != 1 or times.size == 0 or times.dtype.kind not in 'iuf'
            or (times.dtype.kind == 'f' and times.dtype.itemsize < 8)
            or not np.isfinite(times).all() or np.any(times[1:] < times[:-1])
            or isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError('Expected precise chronological timestamps and positive integer batch_size')
    result = np.empty_like(times)
    for start in range(0, len(times), batch_size):
        stop = min(start + batch_size, len(times))
        result[start:stop] = times[stop - 1]
    return result


def chronological_development_split(timestamps):
    """Split ordered events at fixed quantiles into A/B/C/D and old Test.

    A=training (<=45%), B=model selection (45–50%), C=unlabeled reference
    (50–60%), D=development evaluation (60–70%), remainder=old Test.
    Ties remain on the earlier side. Percentages refer to timestamp quantiles;
    actual sample counts can differ. Only timestamps are accepted here.
    """
    times = np.asarray(timestamps)
    if (times.ndim != 1 or times.size == 0 or times.dtype.kind not in 'iuf'
            or (times.dtype.kind == 'f' and times.dtype.itemsize < 8)
            or not np.isfinite(times).all() or np.any(times[1:] < times[:-1])
            or (times.dtype.kind in 'iu' and (np.any(times >= 2**53) or np.any(times <= -(2**53))))):
        raise ValueError('timestamps must be finite, chronological, one-dimensional and precise')
    # Match get_node_classification_data(val_ratio=.05, test_ratio=.5).
    cutoffs = np.quantile(times, [1 - .05 - .5, 1 - .5, .6, .7])
    ends = np.searchsorted(times, cutoffs, side='right').astype(int)
    bounds = np.concatenate(([0], ends, [len(times)]))
    if np.any(np.diff(bounds) == 0):
        raise ValueError('At least one required temporal segment is empty')
    names = ('train', 'selection', 'reference', 'development', 'old_test')
    return {name: slice(int(bounds[i]), int(bounds[i + 1])) for i, name in enumerate(names)}


def paired_development_scores(reference_scores, reference_degrees, reference_times,
                              development_scores, development_degrees, development_times,
                              development_labels, *, model_ready_time, prior_strength):
    """Compare four postprocessors on identical precomputed development scores.

    Original and continuous match the local implementation for cap=20. The
    ECDF is exploratory, uses no reference labels, and is not a probability.
    Labels are read only for development metrics, not to fit any postprocessor.
    This function cannot prove that supplied data or scores are causally valid.
    """
    raw = np.asarray(development_scores)
    degree = np.asarray(development_degrees)
    labels = np.asarray(development_labels)
    if (raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all()
            or degree.ndim != 1 or labels.ndim != 1 or len(raw) != len(degree) or len(raw) != len(labels)
            or set(np.unique(labels)) != {0, 1}):
        raise ValueError('Development scores/degrees/labels must be finite, aligned and binary')
    calibrator = DegreeConditionalECDF(degree_cap=20, prior_strength=prior_strength).fit(
        reference_scores, reference_degrees, reference_times, model_ready_time=model_ready_time)
    ecdf = calibrator.transform(raw, degree, development_times)
    values = {
        'none': raw.copy(),
        'original': raw * np.where(degree < 3, .9, 1.),
        'continuous': raw * (1. - .1 * np.exp(-degree / 3.)),
        'degree_ecdf_prototype': ecdf,
    }
    metrics = {}
    for name, scores in values.items():
        if not np.isfinite(scores).all():
            raise ValueError(f'Non-finite {name} scores')
        precision, recall, _ = precision_recall_curve(labels, scores)
        metrics[name] = {'AUROC': float(roc_auc_score(labels, scores)),
                         'AP': float(average_precision_score(labels, scores)),
                         'PR_AUC': float(auc(recall, precision))}
    return values, metrics
