"""Exploratory raw-score bounded calibration using unlabeled past scores.

This is a local research prototype, not the AHEAD paper or official code.
For finite epsilon >= 0 the absolute adjustment is at most epsilon times the
reference-score IQR. It does not guarantee AUROC improvement.
"""
import numpy as np

from degree_ecdf_prototype import _times, _vector


class BoundedRawCalibration:
    """Bound any provided candidate's adjustment in raw score units.

    The input candidate must already be computed without future information.
    If two raw scores differ by more than 2 * epsilon * reference IQR, their
    order cannot reverse. A zero IQR returns the original scores unchanged.
    """

    def __init__(self, *, epsilon):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.integer, np.floating)):
            raise ValueError('epsilon must be finite and nonnegative')
        self.epsilon = float(epsilon)
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError('epsilon must be finite and nonnegative')
        self._latest_time = None

    def fit(self, reference_scores, timestamps, *, model_ready_time):
        if self._latest_time is not None:
            raise RuntimeError('This reference cannot be silently replaced')
        scores = _vector('reference_scores', reference_scores)
        times = _times('timestamps', timestamps)
        ready = np.asarray(model_ready_time)
        if (len(scores) != len(times) or ready.ndim != 0 or ready.dtype != times.dtype
                or not np.isfinite(ready).item() or np.any(times <= ready)):
            raise ValueError('Reference timestamps must be aligned and strictly after model readiness')
        low, high = np.quantile(scores.astype(np.float64), [.25, .75])
        self._limit = self.epsilon * (high - low)
        if not np.isfinite(self._limit):
            raise ValueError('Calibration limit overflow')
        self._latest_time = times.max().item()
        self._time_dtype = times.dtype
        return self

    def transform(self, raw_scores, candidate_scores, timestamps):
        if self._latest_time is None:
            raise RuntimeError('Fit a reference before transforming scores')
        raw = _vector('raw_scores', raw_scores).astype(np.float64)
        candidate = _vector('candidate_scores', candidate_scores).astype(np.float64)
        times = _times('timestamps', timestamps)
        if (len(raw) != len(candidate) or len(raw) != len(times) or times.dtype != self._time_dtype
                or np.any(times <= self._latest_time)):
            raise ValueError('Queries must be aligned and strictly after the reference')
        with np.errstate(over='ignore', invalid='ignore'):
            adjustment = candidate - raw
        if not np.isfinite(adjustment).all():
            raise ValueError('Calibration adjustment overflow')
        return raw + np.clip(adjustment, -self._limit, self._limit)

    def auc_change_upper_bound(self, raw_scores, labels):
        """Bound absolute AUROC change using labeled close pairs.

        This is an evaluation diagnostic, not an unsupervised calibration or
        a model-selection rule. Never use Test labels to choose a setting.
        For a bound L on each score's movement, only positive-negative pairs
        whose original gap is at most 2L can change their ordering or tie.
        Floating-point padding makes the boundary conservatively wider.
        """
        if self._latest_time is None:
            raise RuntimeError('Fit a reference before computing the bound')
        scores = _vector('raw_scores', raw_scores).astype(np.float64)
        if not np.isfinite(scores).all():
            raise ValueError('raw_scores must remain finite as float64')
        y = np.asarray(labels)
        if (y.ndim != 1 or len(y) != len(scores) or set(np.unique(y)) != {0, 1}):
            raise ValueError('labels must be aligned and contain both binary classes')
        positives = scores[y == 1]
        negatives = np.sort(scores[y == 0])
        # Account for rounding in subtraction, addition, and score updates.
        scale = max(float(np.max(np.abs(scores))), self._limit)
        with np.errstate(over='ignore', invalid='ignore'):
            margin = 2. * self._limit + 4. * np.finfo(np.float64).eps * scale
            low = np.nextafter(positives - margin, -np.inf)
            high = np.nextafter(positives + margin, np.inf)
            for _ in range(3):
                low, high = np.nextafter(low, -np.inf), np.nextafter(high, np.inf)
        left = np.searchsorted(negatives, low, side='left')
        right = np.searchsorted(negatives, high, side='right')
        return int(np.sum(right - left, dtype=np.int64)) / (len(positives) * len(negatives))


class BoundedMedianDegreeCalibration:
    """Exploratory unlabeled median alignment with a bounded raw-score shift.

    Estimate low/high sampled-degree medians on earlier, unlabeled scores.
    Shift low-degree scores toward the high-degree median by at most epsilon
    reference IQR. Sparse groups or zero IQR fall back to the raw scores.
    This assumes typical group-score differences are nuisance effects; it
    can erase useful anomaly signal and does not guarantee better ranking.
    """

    def __init__(self, *, epsilon=0.05, min_group=30):
        if isinstance(min_group, (bool, np.bool_)) or not isinstance(min_group, (int, np.integer)) or min_group < 1:
            raise ValueError('min_group must be a positive integer')
        self.epsilon = BoundedRawCalibration(epsilon=epsilon).epsilon
        self.min_group = int(min_group)
        self._base = None

    @staticmethod
    def _degree(values, expected):
        degree = _vector('degrees', values)
        if (len(degree) != expected or np.any(degree < 0) or np.any(degree > 20)
                or np.any(degree != np.floor(degree))):
            raise ValueError('degrees must be aligned sampled integers in [0, 20]')
        return degree

    def fit(self, reference_scores, reference_degrees, timestamps, *, model_ready_time):
        if self._base is not None:
            raise RuntimeError('This reference cannot be silently replaced')
        scores = _vector('reference_scores', reference_scores).astype(np.float64)
        degree = self._degree(reference_degrees, len(scores))
        base = BoundedRawCalibration(epsilon=self.epsilon).fit(scores, timestamps, model_ready_time=model_ready_time)
        low = degree < 3
        counts = (int(np.count_nonzero(low)), int(np.count_nonzero(~low)))
        if min(counts) < self.min_group:
            shift, reason = 0., 'insufficient_reference_group'
        elif base._limit == 0:
            shift, reason = 0., 'zero_reference_iqr'
        else:
            difference = float(np.median(scores[low]) - np.median(scores[~low]))
            if not np.isfinite(difference):
                raise ValueError('Reference median difference overflow')
            shift, reason = float(np.clip(-difference, -base._limit, base._limit)), 'median_alignment'
        self._base, self.shift, self.reason, self.reference_counts = base, shift, reason, counts
        return self

    def transform(self, raw_scores, degrees, timestamps):
        if self._base is None:
            raise RuntimeError('Fit a reference before transforming scores')
        raw = _vector('raw_scores', raw_scores).astype(np.float64)
        degree = self._degree(degrees, len(raw))
        with np.errstate(over='ignore', invalid='ignore'):
            candidate = raw + np.where(degree < 3, self.shift, 0.)
        return self._base.transform(raw, candidate, timestamps)
