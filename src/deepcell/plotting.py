from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay, \
    confusion_matrix

from deepcell.metrics import CVMetrics


def plot_training_performance(cv_metrics: CVMetrics,
                              which_metric: str,
                              early_stopping_num_epochs: int,
                              num_folds: int):
    """Plots performance as function of epoch over all CV folds

    Args:
        cv_metrics:
            Metrics
        which_metric:
            Which metric to plot
        early_stopping_num_epochs:
             Early stopping parameter. Training stops here, so plot
             truncated here.
        num_folds:
            Number of folds for training
    """
    fig, ax = plt.subplots(nrows=num_folds, ncols=1, figsize=(5, 10))

    for i in range(num_folds):
        if which_metric == 'F1':
            train_metric = cv_metrics.train_metrics[i].f1s
            val_metric = cv_metrics.valid_metrics[i].f1s
        elif which_metric == 'loss':
            train_metric = cv_metrics.train_metrics[i].losses
            val_metric = cv_metrics.valid_metrics[i].losses
        else:
            raise ValueError(
                f'metric {which_metric} not supported. Needs to be '
                f'one of "F1" or "loss"')

        best_epoch = cv_metrics.train_metrics[i].best_epoch

        ax[i].plot(train_metric[:best_epoch + early_stopping_num_epochs],
                   label='train')
        ax[i].plot(val_metric[:best_epoch + early_stopping_num_epochs],
                   label='val')
        ax[i].legend()
        ax[i].set_xlabel('Epoch')
        ax[i].set_ylabel(which_metric)

    plt.show()


def plot_confusion_matrix(classifier_scores: pd.DataFrame, label_col: str,
                          pred_col: str):
    f, axes = plt.subplots(1, 2, figsize=(20, 10), sharey='row')
    for i, normalization in enumerate(('true', 'pred')):
        cm = confusion_matrix(y_true=classifier_scores[label_col],
                              y_pred=classifier_scores[pred_col],
                              normalize=normalization)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['cell', 'not cell'])
        disp.plot(ax=axes[i])
        disp.ax_.set_title(
            f'N={classifier_scores.shape[0]}\n'
            f'normalized by {normalization}')
        disp.ax_.grid(False)

        # Increase font size of values
        for labels_ in disp.text_.ravel():
            labels_.set_fontsize(24)
    plt.show()


def plot_pr_curve(scores: pd.DataFrame):
    scores['y_pred'] = scores['y_pred'].apply(
        lambda x: 'cell' if x else 'not cell')
    scores['y_true'] = scores['y_true'].apply(
        lambda x: 'cell' if x else 'not cell')

    precision, recall, thresholds = precision_recall_curve(
        y_true=scores['y_true'], probas_pred=scores['y_score'],
        pos_label='cell')

    thresholds = np.concatenate([[0], thresholds])

    res = pd.DataFrame(
        {'threshold': np.concatenate([thresholds, thresholds]),
         'value': np.concatenate([precision, recall]),
         'metric': ['precision'] * len(precision) + ['recall'] * len(recall)})

    sns.lineplot(data=res, x='threshold', y='value', hue='metric')
    plt.title('Precision recall curve')
    plt.show()
