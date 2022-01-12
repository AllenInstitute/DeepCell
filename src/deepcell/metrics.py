from typing import Optional, Dict

import numpy as np
import torch
import torchvision
from sklearn.metrics import average_precision_score, f1_score


class TrainingMetrics:
    def __init__(self,
                 n_epochs,
                 best_metric: str,
                 losses: Optional[np.ndarray] = None,
                 best_epoch=-1,
                 best_metric_value: Optional[float] = None,
                 metric_larger_is_better=True,
                 additional_metrics: Optional[Dict] = None):
        if losses is not None:
            losses = np.array(losses.tolist() + [0] * n_epochs)
        else:
            losses = np.zeros(n_epochs)

        self.losses = losses
        self.best_epoch = best_epoch
        self._best_metric = best_metric
        self._metric_larger_is_better = metric_larger_is_better
        self._additional_metrics = {}

        if additional_metrics is not None:
            for metric, val in additional_metrics.items():
                if val is None:
                    self._additional_metrics[metric] = np.zeros(n_epochs)
                else:
                    self._additional_metrics[metric] = \
                        np.array(val.tolist() + [0] * n_epochs)

        if best_metric_value is None:
            if metric_larger_is_better:
                best_metric_value = -float('inf')
            else:
                best_metric_value = float('inf')
        self._best_metric_value = best_metric_value

    @property
    def best_metric_value(self) -> float:
        return self._best_metric_value

    @property
    def best_metric(self):
        return self._best_metric

    def update(self, epoch, loss, best_metric_val,
               additional_metrics: Optional[Dict] = None):
        self.losses[epoch] = loss
        for metric, val in additional_metrics.items():
            self._additional_metrics[metric][epoch] = val

        if self._metric_larger_is_better:
            if best_metric_val > self._best_metric_value:
                self._best_metric_value = best_metric_val
                self.best_epoch = epoch
        else:
            if best_metric_val < self._best_metric_value:
                self._best_metric_value = best_metric_val
                self.best_epoch = epoch

    def to_dict(self, best_epoch: int, to_epoch: int) -> dict:
        """

        Args:
            best_epoch:
                If early stopping, indicates best epoch
            to_epoch:
                If early stopping, truncates performance to `to_epoch`
        Returns:
            Dict of performance
        """
        d = {
            'losses': self.losses[:to_epoch + 1],
            'best_epoch': best_epoch,
            'best_metric': self._best_metric,
            'best_metric_value': self._best_metric_value,
            'metric_larger_is_better': self._metric_larger_is_better
        }
        for metric, vals in self._additional_metrics.items():
            d[metric] = vals[:to_epoch + 1]
        return d


class Metrics:
    def __init__(self):
        self.loss = 0.0

    def update_loss(self, loss, num_batches):
        self.loss += loss / num_batches  # loss*num_batches/N = (Sum of all l)/N
    
    def update_outputs(self, y_out, y_true):
        raise NotImplementedError


class ClassificationMetrics(Metrics):
    def __init__(self):
        super().__init__()

        self.TP = 0
        self.FP = 0
        self.FN = 0
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

    def _increment_TP(self, y_pred, y_true):
        self.TP += ((y_true == 1) & (y_pred == 1)).sum().item()

    def _increment_FP(self, y_pred, y_true):
        self.FP += ((y_true == 0) & (y_pred == 1)).sum().item()

    def _increment_FN(self, y_pred, y_true):
        self.FN += ((y_true == 1) & (y_pred == 0)).sum().item()


class LocalizationMetrics(Metrics):
    def __init__(self):
        super().__init__()
        self._bounding_box_trues = []
        self._bounding_box_preds = []
    
    @property
    def IOU(self):
        # TODO consider making this vectorized
        ious = np.zeros(len(self._bounding_box_trues))

        for i in range(len(self._bounding_box_trues)):
            x1_1, y1_1, w_1, h_1 = self._bounding_box_trues[i]
            x1_2, y1_2, w_2, h_2 = self._bounding_box_preds[i]

            x2_1 = x1_1 + w_1
            y2_1 = y1_1 + h_1

            x2_2 = x2_1 + w_2
            y2_2 = y2_1 + h_2

            bb_true = torch.tensor([[x1_1, y1_1, x2_1, y2_1]])
            bb_pred = torch.tensor([[x1_2, y1_2, x2_2, y2_2]])

            iou = torchvision.ops.box_iou(bb_true, bb_pred)
            ious[i] = iou.item()

        return ious.mean()

    def update_outputs(self, y_out, y_true):
        bbs_preds = y_out.detach().cpu().numpy().tolist()
        bb_true = y_true.detach().cpu().numpy().tolist()
        self._bounding_box_preds += bbs_preds
        self._bounding_box_trues += bb_true



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




