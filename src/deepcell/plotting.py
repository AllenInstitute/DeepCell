from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

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


def pr_curve(probas_pred, y_true):
    """Plots PR curve

    Args:
        probas_pred:
            Classification probabilities
        y_true:
            True labels
    """
    precision, recall, thresholds = precision_recall_curve(
        probas_pred=probas_pred, y_true=y_true)
    thresholds = thresholds.tolist()
    thresholds.append(1.0)
    fig, ax = plt.subplots()
    ax.plot(thresholds, precision, label='precision')
    ax.plot(thresholds, recall, label='recall')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric')
    ax.set_title('Precision-recall curve')
    ax.legend()

    return fig
