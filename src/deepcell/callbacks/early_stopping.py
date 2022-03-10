from typing import Optional, Dict

import numpy as np

from deepcell.callbacks.base_callback import Callback


class EarlyStopping(Callback):
    def __init__(self,
                 time_since_best_epoch=0,
                 best_epoch:int = -1,
                 best_metric: str = 'f1',
                 best_metric_value: Optional[float] = None,
                 metric_larger_is_better: bool = True,
                 patience: int = 0):
        """
        Early stopping callback
        Args:
            time_since_best_epoch
                Number of epochs since previous best metric value
            best_epoch
                If provided will set best_epoch to this
                Useful if continuing training
            best_metric
                Metric to use for early stopping
            best_metric_value
                Best metric value so far.
            metric_larger_is_better
                Whether a larger value of the metric is better
            patience
                Number of epochs before early stopping is initiated
        """
        self._time_since_best_epoch = time_since_best_epoch
        self._best_epoch = best_epoch
        self._best_metric = best_metric
        self._metric_larger_is_better = metric_larger_is_better
        self._patience = patience

        if best_metric_value is None:
            if best_metric == 'f1':
                best_metric_value = -float('inf')
            elif best_metric == 'loss':
                best_metric_value = float('inf')
            else:
                raise ValueError(f'Unsupported best_metric. Needs to be '
                                 f'either "f1" or "loss"')
        self._best_metric_value = best_metric_value

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def best_metric_value(self) -> float:
        return self._best_metric_value

    @property
    def time_since_best_epoch(self) -> int:
        return self._time_since_best_epoch

    @property
    def patience(self):
        return self._patience

    @property
    def best_metric(self):
        return self._best_metric

    @time_since_best_epoch.setter
    def time_since_best_epoch(self, value):
        self._time_since_best_epoch = value

    @best_metric.setter
    def best_metric(self, value):
        self._best_metric = value

    @best_metric_value.setter
    def best_metric_value(self, value):
        self._best_metric_value = value

    @best_epoch.setter
    def best_epoch(self, value):
        self._best_epoch = value

    def on_epoch_end(self, epoch: int, metrics: Dict[str, np.ndarray]):
        if self._best_metric == 'f1':
            metric = metrics['val_f1'][epoch]
        else:
            metric = metrics['val_loss'][epoch]

        if self._metric_larger_is_better:
            if metric > self._best_metric_value:
                self._best_metric_value = metric
                self._best_epoch = epoch
        else:
            if metric < self._best_metric_value:
                self._best_metric_value = metric
                self._best_epoch = epoch

    def to_dict(self) -> dict:
        d = {
            'best_epoch': self._best_epoch,
            'best_metric': self._best_metric,
            'best_metric_value': self._best_metric_value,
            'metric_larger_is_better': self._metric_larger_is_better
        }
        return d
