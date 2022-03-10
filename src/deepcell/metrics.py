from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score


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
        self._metrics = []

    def update(self, metrics: Dict[str, np.ndarray], best_epoch=None):
        if best_epoch is None:
            best_epoch = len(metrics['loss'])

        best_epoch_metric_vals = {}
        for k in metrics:
            best_epoch_metric_vals[k] = metrics[k][best_epoch]
        self._metrics.append(best_epoch_metric_vals)

    @property
    def metrics(self):
        train_loss = np.array([x['loss'] for x in self._metrics])
        train_loss = train_loss.mean()

        valid_loss = np.array([x['val_loss'] for x in self._metrics])
        valid_loss = valid_loss.mean()

        return {
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }




