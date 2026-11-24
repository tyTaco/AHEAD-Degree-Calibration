"""Experimental label-informed selection between two fixed score vectors.

Inputs must be from a chronological validation interval, never Test labels.
Block bootstrap sensitivity is not a distribution-free confidence guarantee.
The caller must ensure that scores were available by the selection time.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class SelectionResult:
    use_candidate: bool
    observed_delta: float
    bootstrap_lower: float
    valid_replicates: int
    requested_replicates: int


def select_with_block_bootstrap(labels, baseline, candidate, *, block_size=200,
                                replicates=300, lower_quantile=.05, seed=0):
    """Select candidate only if the bootstrap lower percentile exceeds zero.

    Resamples consecutive, non-overlapping event blocks. A real time series
    need not satisfy bootstrap assumptions; results are an exploratory
    sensitivity analysis, not a promised risk bound or an unsupervised method.
    """
    y, base, alt = map(np.asarray, (labels, baseline, candidate))
    if (any(x.ndim != 1 or not len(x) for x in (y, base, alt))
            or len(y) != len(base) or len(y) != len(alt)
            or set(np.unique(y)) != {0, 1}
            or base.dtype.kind not in 'iuf' or alt.dtype.kind not in 'iuf'
            or not np.isfinite(base).all() or not np.isfinite(alt).all()):
        raise ValueError('Expected aligned binary labels and finite scores')
    if (isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)) or block_size < 1
            or isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates < 20
            or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
            or isinstance(lower_quantile, bool)
            or not isinstance(lower_quantile, (int, float, np.integer, np.floating))
            or not 0 < lower_quantile < .5):
        raise ValueError('Invalid bootstrap settings')
    blocks = [np.arange(start, min(start + block_size, len(y))) for start in range(0, len(y), block_size)]
    if len(blocks) < 2:
        raise ValueError('At least two time blocks are required')
    observed = float(roc_auc_score(y, alt) - roc_auc_score(y, base))
    generator = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        ids = np.concatenate([blocks[i] for i in generator.integers(len(blocks), size=len(blocks))])
        if len(np.unique(y[ids])) != 2:
            continue
        deltas.append(float(roc_auc_score(y[ids], alt[ids]) - roc_auc_score(y[ids], base[ids])))
    if len(deltas) < .9 * replicates:
        raise ValueError('Too few bootstrap samples contain both label classes')
    lower = float(np.quantile(deltas, lower_quantile))
    return SelectionResult(observed > 0 and lower > 0, observed, lower, len(deltas), int(replicates))
