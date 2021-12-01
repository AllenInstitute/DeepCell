from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score


class TrainingMetrics:
    def __init__(self, n_epochs,
                 losses: Optional[np.ndarray] = None,
                 f1s: Optional[np.ndarray] = None,
                 best_epoch=-1,
                 best_metric='f1',
                 best_metric_value: Optional[float] = None,
                 metric_larger_is_better=True):
        """
        Container for training metrics
        Args:
            n_epochs:
                Number of training epochs
            losses:
                If provided, will prepend these losses.
                Useful if continuing training
            f1s:
                If provided, will prepend these f1s.
                Useful if continuing training
            best_epoch
                If provided will set best_epoch to this
                Useful if continuing training
            best_metric
                Metric to use for early stopping
            best_metric_value
                Best metric value so far.
                Used for early stopping
            metric_larger_is_better
                Whether a larger value of the metric is better
        """
        if losses is not None:
            losses = np.array(losses.tolist() + [0] * n_epochs)
        else:
            losses = np.zeros(n_epochs)

        if f1s is not None:
            f1s = np.array(f1s.tolist() + [0] * n_epochs)
        else:
            f1s = np.zeros(n_epochs)

        self.losses = losses
        self.f1s = f1s
        self.best_epoch = best_epoch
        self._best_metric = best_metric
        self._metric_larger_is_better = metric_larger_is_better

        if best_metric_value is None:
            if best_metric == 'f1':
                best_metric_value = -float('inf')
            elif best_metric == 'loss':
                best_metric_value = float('inf')
            else:
                raise ValueError(f'Unsupported best_metric. Needs to be '
                                 f'either "f1" or "loss"')
        self._best_metric_value = best_metric_value

    def update(self, epoch, loss, f1):
        self.losses[epoch] = loss
        self.f1s[epoch] = f1

        if self._best_metric == 'f1':
            metric = f1
        else:
            metric = loss

        if self._metric_larger_is_better:
            if metric > self._best_metric_value:
                self._best_metric_value = metric
                self.best_epoch = epoch
        else:
            if metric < self._best_metric_value:
                self._best_metric_value = metric
                self.best_epoch = epoch

    @property
    def best_metric_value(self) -> float:
        return self._best_metric_value

    def to_dict(self, best_epoch: int) -> dict:
        d = {
            'f1s': self.f1s[:best_epoch+1],
            'losses': self.losses[:best_epoch+1],
            'best_epoch': best_epoch,
            'best_metric': self._best_metric,
            'best_metric_value': self._best_metric_value,
            'metric_larger_is_better': self._metric_larger_is_better
        }
        return d


class Metrics:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.loss = 0.0
        self.y_scores = []
        self.y_trues = []

    @property
    def F1(self):
        y_pred = np.array(self.y_scores) > 0.5
        y_pred = y_pred.astype('int')
        f1 = f1_score(y_true=self.y_trues, y_pred=y_pred,
                      zero_division=0)
        return f1

    @property
    def AUPR(self):
        return average_precision_score(y_true=self.y_trues, y_score=self.y_scores)

    def update_accuracies(self, y_true, y_out=None, y_score=None, threshold=0.5):
        if y_out is not None and y_score is not None:
            raise ValueError('Supply either logits or score')
        if y_out is None and y_score is None:
            raise ValueError('Supply either logits or score')

        if y_score is None:
            y_score = torch.sigmoid(y_out)
        y_pred = y_score > threshold

        self._increment_TP(y_pred=y_pred, y_true=y_true)
        self._increment_FP(y_pred=y_pred, y_true=y_true)
        self._increment_FN(y_pred=y_pred, y_true=y_true)

    def update_outputs(self, y_out, y_true):
        preds = torch.sigmoid(y_out).detach().cpu().numpy().tolist()
        y_true = y_true.detach().cpu().numpy().tolist()
        self.y_scores += preds
        self.y_trues += y_true

    def update_loss(self, loss, num_batches):
        self.loss += loss / num_batches  # loss*num_batches/N = (Sum of all l)/N

    def _increment_TP(self, y_pred, y_true):
        self.TP += ((y_true == 1) & (y_pred == 1)).sum().item()

    def _increment_FP(self, y_pred, y_true):
        self.FP += ((y_true == 0) & (y_pred == 1)).sum().item()

    def _increment_FN(self, y_pred, y_true):
        self.FN += ((y_true == 1) & (y_pred == 0)).sum().item()


class CVMetrics:
    def __init__(self, n_splits):
        self.n_splits = n_splits
        self.train_metrics = []
        self.valid_metrics = []

    def update(self, train_metrics, valid_metrics):
        self.train_metrics.append(train_metrics)
        self.valid_metrics.append(valid_metrics)

    @property
    def metrics(self):
        train_loss = sum([x.losses[x.best_epoch] for x in self.train_metrics]) / self.n_splits
        valid_loss = sum([x.losses[x.best_epoch] for x in self.valid_metrics]) / self.n_splits

        return {
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }




