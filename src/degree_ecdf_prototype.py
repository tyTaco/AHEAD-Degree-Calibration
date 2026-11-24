"""Research-only unlabeled, degree-conditioned empirical score percentiles.

This is not the paper's degree calibration and is not integrated into AHEAD.
Scores used as reference must themselves be produced by a valid, predeclared
time protocol. This module does not load data, labels, models, or checkpoints.
"""
import numpy as np


def _vector(name, values):
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f'{name} must be a nonempty one-dimensional array')
    if result.dtype.kind not in 'iuf' or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite real numbers')
    return result


def _times(name, values):
    values = _vector(name, values)
    if values.dtype.kind == 'f' and values.dtype.itemsize < 8:
        raise ValueError(f'{name} must use float64 or integer timestamps')
    return values


def _degree_groups(degrees, cap):
    degrees = _vector('degrees', degrees)
    if np.any(degrees != np.floor(degrees)) or np.any(degrees < 0) or np.any(degrees > cap):
        raise ValueError('degrees must be integer values between zero and degree_cap')
    # Zero, one-or-two, three-to-cap-minus-one, and saturated count.
    return np.searchsorted(np.array([1, 3, cap]), degrees, side='right')


def _midrank_ecdf(sorted_reference, values):
    """Smoothed empirical percentile, with equal values receiving midranks."""
    n = len(sorted_reference)
    if n == 0:
        raise ValueError('reference distribution cannot be empty')
    left = np.searchsorted(sorted_reference, values, side='left')
    right = np.searchsorted(sorted_reference, values, side='right')
    return (left + .5 * (right - left) + .5) / (n + 1.)


class DegreeConditionalECDF:
    """Shrink group percentiles towards a pooled unlabeled reference.

    For a group with n references and prior_strength λ, w=n/(n+λ).
    Its output is w*F_group(s)+(1-w)*F_pool(s); empty groups use F_pool.
    The declared model-ready time must precede every reference timestamp, and
    every query timestamp must be strictly later than the reference. The caller
    must independently verify that the declared model-ready time accounts for
    all data used for model fitting/selection and that scores are causal.
    Output is a percentile, not an anomaly probability, and can hide genuine
    degree-associated anomalies.
    """

    def __init__(self, *, degree_cap, prior_strength):
        if isinstance(degree_cap, bool) or not isinstance(degree_cap, (int, np.integer)) or degree_cap <= 3:
            raise ValueError('degree_cap must be an integer greater than 3')
        if isinstance(prior_strength, bool) or not np.isscalar(prior_strength):
            raise ValueError('prior_strength must be a finite nonnegative number')
        try:
            strength = float(prior_strength)
        except (TypeError, ValueError) as exc:
            raise ValueError('prior_strength must be a finite nonnegative number') from exc
        if not np.isfinite(strength) or strength < 0:
            raise ValueError('prior_strength must be a finite nonnegative number')
        self.degree_cap = int(degree_cap)
        self.prior_strength = strength
        self._pooled = None

    def fit(self, scores, degrees, timestamps, *, model_ready_time):
        if self._pooled is not None:
            raise RuntimeError('This fitted reference cannot be silently replaced')
        scores = _vector('scores', scores)
        groups = _degree_groups(degrees, self.degree_cap)
        timestamps = _times('timestamps', timestamps)
        if len(scores) != len(groups) or len(scores) != len(timestamps):
            raise ValueError('scores, degrees and timestamps must be aligned')
        ready = np.asarray(model_ready_time)
        if (ready.ndim != 0 or ready.dtype != timestamps.dtype
                or not np.isfinite(ready).item()):
            raise ValueError('model_ready_time must be finite and share the timestamp dtype')
        if np.any(timestamps <= ready):
            raise ValueError('All reference timestamps must be strictly later than model_ready_time')
        self._pooled = np.sort(scores.copy())
        self._groups = tuple(np.sort(scores[groups == group].copy()) for group in range(4))
        self._latest_time = timestamps.max().item()
        self._time_dtype = timestamps.dtype
        return self

    def transform(self, scores, degrees, timestamps):
        if self._pooled is None:
            raise RuntimeError('Fit a reference before transforming scores')
        scores = _vector('scores', scores)
        groups = _degree_groups(degrees, self.degree_cap)
        timestamps = _times('timestamps', timestamps)
        if len(scores) != len(groups) or len(scores) != len(timestamps):
            raise ValueError('scores, degrees and timestamps must be aligned')
        if timestamps.dtype != self._time_dtype:
            raise ValueError('Query and reference timestamps must have the same dtype')
        if np.any(timestamps <= self._latest_time):
            raise ValueError('Every query must be strictly later than the reference cutoff')
        pooled = _midrank_ecdf(self._pooled, scores)
        result = pooled.copy()
        for group, reference in enumerate(self._groups):
            mask = groups == group
            if reference.size and np.any(mask):
                w = reference.size / (reference.size + self.prior_strength)
                result[mask] = (w * _midrank_ecdf(reference, scores[mask])
                                + (1. - w) * pooled[mask])
        return result
