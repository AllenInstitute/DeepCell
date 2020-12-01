import numpy as np
import torch


class TrainingMetrics:
    def __init__(self, n_epochs):
        self.losses = np.zeros(n_epochs)
        self.precisions = np.zeros(n_epochs)
        self.recalls = np.zeros(n_epochs)
        self.f1s = np.zeros(n_epochs)

    def update(self, epoch, loss, precision, recall, f1):
        self.losses[epoch] = loss
        self.precisions[epoch] = precision
        self.recalls[epoch] = recall
        self.f1s[epoch] = f1

    def truncate_to_epoch(self, epoch):
        self.losses = self.losses[:epoch]
        self.precisions = self.precisions[:epoch]
        self.recalls = self.recalls[:epoch]
        self.f1s = self.f1s[:epoch]


class Metrics:
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.loss = 0.0

    @property
    def precision(self):
        try:
            res = self.TP / (self.TP + self.FP)
        except ZeroDivisionError:
            res = 0.0
        return res

    @property
    def recall(self):
        try:
            res = self.TP / (self.TP + self.FN)
        except ZeroDivisionError:
            res = 0.0
        return res

    @property
    def F1(self):
        precision = self.precision
        recall = self.recall

        try:
            res = 2 * precision * recall / (recall + precision)
        except ZeroDivisionError:
            res = 0.0
        return res

    def update_accuracies(self, y_true, y_out=None, y_score=None):
        if y_out is not None and y_score is not None:
            raise ValueError('Supply either logits or score')
        if y_out is None and y_score is None:
            raise ValueError('Supply either logits or score')

        if y_score is None:
            y_score = torch.sigmoid(y_out)
        y_pred = y_score > .5

        self._increment_TP(y_pred=y_pred, y_true=y_true)
        self._increment_FP(y_pred=y_pred, y_true=y_true)
        self._increment_FN(y_pred=y_pred, y_true=y_true)

    def update_loss(self, loss, num_batches):
        self.loss += loss / num_batches  # loss*num_batches/N = (Sum of all l)/N

    def _increment_TP(self, y_pred, y_true):
        self.TP += ((y_true == 1) & (y_pred == 1)).sum().item()

    def _increment_FP(self, y_pred, y_true):
        self.FP += ((y_true == 0) & (y_pred == 1)).sum().item()

    def _increment_FN(self, y_pred, y_true):
        self.FN += ((y_true == 1) & (y_pred == 0)).sum().item()


class CVMetrics:
    def __init__(self, n_splits, n_epochs):
        self.n_splits = n_splits
        self.min_n_epoch = float('inf')
        self.train_metrics = []
        self.valid_metrics = []

    def update(self, train_metrics, valid_metrics):
        self.train_metrics.append(train_metrics)
        self.valid_metrics.append(valid_metrics)
        self.min_n_epoch = min(len(train_metrics.losses), self.min_n_epoch)

    @property
    def metrics(self):
        valid_precision = sum([x.precisions[-1] for x in self.valid_metrics]) / self.n_splits
        valid_recall = sum([x.recalls[-1] for x in self.valid_metrics]) / self.n_splits

        train_f1 = sum([x.f1s[-1] for x in self.train_metrics]) / self.n_splits
        valid_f1 = sum([x.f1s[-1] for x in self.valid_metrics]) / self.n_splits

        return {
            'valid_precision': valid_precision,
            'valid_recall': valid_recall,
            'train_f1': train_f1,
            'valid_f1': valid_f1,
        }

    def _get_running_metrics(self):
        for x in self.train_metrics:
            x.truncate_to_epoch(epoch=self.min_n_epoch)
        for x in self.valid_metrics:
            x.truncate_to_epoch(epoch=self.min_n_epoch)

        train_loss = np.zeros(int(self.min_n_epoch))
        valid_loss = np.zeros(int(self.min_n_epoch))

        train_f1 = np.zeros(int(self.min_n_epoch))
        valid_f1 = np.zeros(int(self.min_n_epoch))

        for m in self.train_metrics:
            train_loss += m.losses
            train_f1 += m.f1s
        train_loss /= self.n_splits
        train_f1 /= self.n_splits

        for m in self.valid_metrics:
            valid_loss += m.losses
            valid_f1 += m.f1s
        valid_loss /= self.n_splits
        valid_f1 /= self.n_splits

        return {
            'loss': (train_loss, valid_loss),
            'f1': (train_f1, valid_f1)
        }




