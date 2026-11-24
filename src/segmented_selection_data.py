"""Experimental metadata/selection-label reader; not a training entry point.

Only the selection label column is materialized when requested. The CSV reader
may scan or buffer other bytes, and all timestamps are used for split quantiles.
This does not provide an OS-level access-control guarantee.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionData:
    metadata: pd.DataFrame
    train: slice
    selection: slice
    selection_labels: np.ndarray | None


def model_data(selection_data):
    """Adapt an experimental split to the existing model's Data interface.

    Return train, B, and the A+B sampler prefix. Only B can contain labels.
    Future metadata remains present in selection_data for quantile calculation.
    """
    from utils.DataLoader import Data

    def adapt(frame, labels):
        return Data(src_node_ids=frame.u.to_numpy(dtype=np.int64),
                    dst_node_ids=frame.i.to_numpy(dtype=np.int64),
                    node_interact_times=frame.ts.to_numpy(dtype=np.float64),
                    edge_ids=frame.idx.to_numpy(dtype=np.int64), labels=labels)

    train = adapt(selection_data.metadata.iloc[selection_data.train], None)
    selection = adapt(selection_data.metadata.iloc[selection_data.selection],
                      selection_data.selection_labels)
    prefix = adapt(selection_data.metadata.iloc[:selection_data.selection.stop], None)
    return train, selection, prefix


def load_selection_data(csv_path, *, val_ratio, test_ratio, use_selection_labels):
    """Read event metadata and, optionally, only the chronological B labels.

    Train is <= first timestamp quantile; B is > first and <= second, matching
    get_node_classification_data. Metadata includes future events and labels are
    not attached to it. This helper does not load features or train a model.
    """
    if not isinstance(use_selection_labels, bool):
        raise ValueError('use_selection_labels must be a boolean')
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError('Invalid chronological split ratios')
    path = Path(csv_path)
    frame = pd.read_csv(path, usecols=['u', 'i', 'ts', 'idx'])
    times = frame['ts'].to_numpy(dtype=np.float64)
    if (not len(times) or not np.isfinite(times).all() or np.any(np.abs(times) >= 2**53)
            or np.any(np.diff(times) < 0)):
        raise ValueError('Timestamps must be nonempty, finite, precise, and sorted')
    first, second = np.quantile(times, [1 - val_ratio - test_ratio, 1 - test_ratio])
    train_end, selection_end = np.searchsorted(times, [first, second], side='right')
    if train_end == 0 or selection_end == train_end or selection_end == len(times):
        raise ValueError('Chronological segments must be nonempty')
    labels = None
    if use_selection_labels:
        labels = pd.read_csv(path, usecols=['label'], skiprows=range(1, int(train_end) + 1),
                             nrows=int(selection_end - train_end))['label'].to_numpy()
        if len(labels) != selection_end - train_end or set(np.unique(labels)) != {0, 1}:
            raise ValueError('Selection labels must contain both binary classes')
    return SelectionData(frame, slice(0, int(train_end)),
                         slice(int(train_end), int(selection_end)), labels)
