"""Exploratory label-informed low-degree calibration; not an unsupervised method.

The caller must provide earlier selection labels and their justified release
time. A timestamp supplied here cannot prove when a real label became known.
Never fit on Test labels. This prototype is not the AHEAD paper or official code.
Bitcoin-OTC historical labels have been found to depend on later interactions;
do not use this prototype on real data until label availability is audited.
"""
import numpy as np

from degree_ecdf_prototype import _degree_groups, _times, _vector


def evaluate_paired_calibrations(labels, scores_by_method, event_times, score_available_times,
                                 label_ready_times, *, selection_decision_time, evaluation_time, batch_size):
    """Report fixed paired scores; declared times and event alignment require independent audit.

    This does not freeze models, establish label provenance, or select a method.
    The caller must not use evaluation labels or results to select a method.
    Batch boundaries are assumed to start at the first supplied event.
    """
    from degree_development import batch_completion_times
    from train_research import research_ranking_metrics

    events = _times('event_times', event_times)
    score_ready = _times('score_available_times', score_available_times)
    label_ready = _times('label_ready_times', label_ready_times)
    y = _vector('labels', labels)
    decision, evaluation = np.asarray(selection_decision_time), np.asarray(evaluation_time)
    if (len({len(events), len(score_ready), len(label_ready), len(y)}) != 1
            or set(np.unique(y)) != {0, 1} or np.any(events[1:] < events[:-1])
            or score_ready.dtype != events.dtype or label_ready.dtype != events.dtype
            or decision.ndim != 0 or evaluation.ndim != 0
            or decision.dtype != events.dtype or evaluation.dtype != events.dtype
            or not np.isfinite(decision).item() or not np.isfinite(evaluation).item()
            or np.any(events <= decision) or np.any(label_ready < events)
            or np.any(score_ready > evaluation) or np.any(label_ready > evaluation)):
        raise ValueError('Evaluation events and declared availability must be aligned and chronological')
    if np.any(score_ready < batch_completion_times(events, batch_size)):
        raise ValueError('Scores cannot be available before the declared batch completes')
    names = ('none', 'original', 'unsupervised', 'supervised')
    if not isinstance(scores_by_method, dict) or set(scores_by_method) != set(names):
        raise ValueError('Exactly four paired method score arrays are required')
    metrics = {}
    for name in names:
        scores = _vector(name, scores_by_method[name])
        if len(scores) != len(y):
            raise ValueError('Each method must score the same events')
        metrics[name] = research_ranking_metrics(y, scores)
    return {'metrics': metrics, **{
        f'delta_vs_{baseline}': {
            name: {metric: value - metrics[baseline][metric] for metric, value in values.items()}
            for name, values in metrics.items()}
        for baseline in ('none', 'original')}}


class HistoricalRiskCalibration:
    """Choose +alpha or -alpha for low sampled degree using historical labels.

    The empirical low/high class rates select the direction; a tie selects no
    adjustment. The fixed magnitude is not fitted here. This method assumes
    finite nonnegative raw scores and that declared labels were actually known.
    """

    def __init__(self, *, alpha=.1, degree_cap=20):
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
            raise ValueError('alpha must be finite and in [0, 1)')
        self.alpha = float(alpha)
        if not np.isfinite(self.alpha) or not 0 <= self.alpha < 1:
            raise ValueError('alpha must be finite and in [0, 1)')
        if isinstance(degree_cap, bool) or not isinstance(degree_cap, (int, np.integer)) or degree_cap <= 3:
            raise ValueError('degree_cap must be an integer greater than 3')
        self.degree_cap = int(degree_cap)
        self._ready_at = None

    def fit(self, degrees, labels, event_times, *, labels_ready_at):
        if self._ready_at is not None:
            raise RuntimeError('This fitted historical reference cannot be silently replaced')
        groups = _degree_groups(degrees, self.degree_cap)
        labels = _vector('labels', labels)
        times = _times('event_times', event_times)
        ready = np.asarray(labels_ready_at)
        if (len(groups) != len(labels) or len(groups) != len(times) or set(np.unique(labels)) != {0, 1}
                or np.any(times[1:] < times[:-1]) or ready.ndim != 0 or ready.dtype != times.dtype
                or not np.isfinite(ready).item() or np.any(times > ready)):
            raise ValueError('Historical binary labels must be aligned, ordered and available by labels_ready_at')
        low = groups < 2
        if not low.any() or low.all():
            raise ValueError('Both low- and high-degree groups are required')
        self.low_rate = float(labels[low].mean())
        self.high_rate = float(labels[~low].mean())
        self.low_factor = 1. + self.alpha * np.sign(self.low_rate - self.high_rate)
        self._ready_at = ready.item()
        self._time_dtype = times.dtype
        return self

    def transform(self, raw_scores, degrees, score_available_times):
        if self._ready_at is None:
            raise RuntimeError('Fit the historical reference first')
        scores = _vector('raw_scores', raw_scores)
        groups = _degree_groups(degrees, self.degree_cap)
        available = _times('score_available_times', score_available_times)
        if (len(scores) != len(groups) or len(scores) != len(available)
                or np.any(scores < 0) or available.dtype != self._time_dtype
                or np.any(available <= self._ready_at)):
            raise ValueError('Scores must be nonnegative, aligned and available strictly after historical labels')
        with np.errstate(over='ignore', invalid='ignore'):
            adjusted = scores.astype(np.float64) * np.where(groups < 2, self.low_factor, 1.)
        if not np.isfinite(adjusted).all():
            raise ValueError('Adjusted scores overflow')
        return adjusted


class HistoricalPairwiseCalibration:
    """Choose a fixed low-degree multiplier by past labeled AUROC.

    This label-supervised prototype compares 1, 1-alpha and 1+alpha on a
    caller-provided historical interval; ties retain 1 where possible. It is
    not the paper or official implementation and may overfit or fail under
    drift. Declared label release times must be independently substantiated.
    Do not use Test labels for selection; no real-data use until the audit ends.
    """

    def __init__(self, *, alpha=.1, degree_cap=20):
        validator = HistoricalRiskCalibration(alpha=alpha, degree_cap=degree_cap)
        self.alpha, self.degree_cap = validator.alpha, validator.degree_cap
        self._decision_time = None

    def fit(self, scores, degrees, labels, event_times, score_available_times, label_ready_times,
            *, model_ready_time, decision_time, batch_size):
        from sklearn.metrics import roc_auc_score
        from degree_development import batch_completion_times

        if self._decision_time is not None:
            raise RuntimeError('This fitted selection cannot be silently replaced')
        raw = _vector('scores', scores).astype(np.float64)
        groups = _degree_groups(degrees, self.degree_cap)
        y = _vector('labels', labels)
        times = _times('event_times', event_times)
        score_ready = _times('score_available_times', score_available_times)
        labels_ready = _times('label_ready_times', label_ready_times)
        model_ready, decision = np.asarray(model_ready_time), np.asarray(decision_time)
        if (len({len(raw), len(groups), len(y), len(times), len(score_ready), len(labels_ready)}) != 1
                or not np.isfinite(raw).all() or np.any(raw < 0) or set(np.unique(y)) != {0, 1}
                or np.any(times[1:] < times[:-1]) or score_ready.dtype != times.dtype
                or labels_ready.dtype != times.dtype or model_ready.ndim != 0 or decision.ndim != 0
                or model_ready.dtype != times.dtype or decision.dtype != times.dtype
                or not np.isfinite(model_ready).item() or not np.isfinite(decision).item()
                or np.any(times <= model_ready) or np.any(score_ready < times)
                or np.any(labels_ready < times) or np.any(score_ready > decision)
                or np.any(labels_ready > decision)):
            raise ValueError('Historical scores and binary labels must be aligned and declared available')
        if np.any(score_ready < batch_completion_times(times, batch_size)):
            raise ValueError('Historical scores cannot be available before their declared batch completes')
        self.aurocs = {}
        best_factor, best_auc = 1., -np.inf
        for factor in (1., 1. - self.alpha, 1. + self.alpha):
            with np.errstate(over='ignore', invalid='ignore'):
                adjusted = raw * np.where(groups < 2, factor, 1.)
            if not np.isfinite(adjusted).all():
                raise ValueError('Adjusted scores overflow')
            auc = float(roc_auc_score(y, adjusted))
            self.aurocs[factor] = auc
            if auc > best_auc:
                best_factor, best_auc = factor, auc
        self.factor = best_factor
        self._decision_time = decision.item()
        self._time_dtype = times.dtype
        return self

    def transform(self, scores, degrees, score_available_times, *, event_times, batch_size):
        """Apply after selection and after the declared score batches finish."""
        from degree_development import batch_completion_times

        if self._decision_time is None:
            raise RuntimeError('Fit the historical selection first')
        raw = _vector('scores', scores).astype(np.float64)
        groups = _degree_groups(degrees, self.degree_cap)
        available = _times('score_available_times', score_available_times)
        events = _times('event_times', event_times)
        if (len({len(raw), len(groups), len(available), len(events)}) != 1 or not np.isfinite(raw).all()
                or np.any(raw < 0) or available.dtype != self._time_dtype or events.dtype != self._time_dtype
                or np.any(events[1:] < events[:-1]) or np.any(events <= self._decision_time)
                or np.any(available < events)):
            raise ValueError('Events must be ordered and strictly after the decision, with scores available afterwards')
        if np.any(available < batch_completion_times(events, batch_size)):
            raise ValueError('Scores cannot be available before their declared batch completes')
        with np.errstate(over='ignore', invalid='ignore'):
            adjusted = raw * np.where(groups < 2, self.factor, 1.)
        if not np.isfinite(adjusted).all():
            raise ValueError('Adjusted scores overflow')
        return adjusted
