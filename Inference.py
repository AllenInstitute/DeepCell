from typing import Tuple

import numpy as np
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from Metrics import Metrics
from RoiDataset import RoiDataset


def inference(model: torch.nn.Module, test_loader: DataLoader, checkpoint_path,
              has_labels=True, threshold=0.5, ensemble=True,
              cv_fold=None, use_cuda=True,
              tta_num_iters=0) -> Tuple[Metrics, pd.DataFrame]:
    """
    Args:
        model:
            torch.nn.Module
        test_loader:
            Dataloader
        checkpoint_path:
            Path to model checkpoints
        has_labels:
            Whether test dataset has labels
        threshold:
            Probability threshold for classification
        ensemble:
            Whether checkpoint_path points to an ensemble of checkpoints to use
            If so, inference will be made once per model and averaged
        cv_fold:
            Whether to use a specific model cv_fold from checkpoint_path
        use_cuda:
            Should be True if on GPU
        tta_num_iters:
            Number of times to perform test-time augmentation (default 0)

    Returns:
        Tuple of Metrics object and dataframe of predictions
    """
    if not ensemble and cv_fold is None:
        raise ValueError('If not using ensemble, need to give cv_fold')
    if cv_fold is not None and ensemble:
        raise ValueError('Ensemble should be false if passing in cv_fold')

    if tta_num_iters == 1:
        raise ValueError('test_time_augmentation_num_iters == 1 does not '
                         'make sense. Either use 0 to disable or a number > '
                         '1.')

    dataset: RoiDataset = test_loader.dataset
    metrics = Metrics()

    if ensemble:
        models = os.listdir(checkpoint_path)
        models = [model for model in models if model != 'model_init.pt']
    else:
        models = [f'{cv_fold}_model.pt']

    num_iters = 1 if tta_num_iters == 0 else tta_num_iters

    y_scores = np.zeros((len(models), len(dataset), num_iters))
    y_preds = np.zeros((len(models), len(dataset), num_iters))

    for i, model_checkpoint in enumerate(models):
        state_dict = torch.load(f'{checkpoint_path}/{model_checkpoint}')
        model.load_state_dict(state_dict)

        model.eval()
        prev_start = 0

        for iter in range(num_iters):
            for data, _ in test_loader:
                if use_cuda:
                    data = data.cuda()

                with torch.no_grad():
                    output = model(data)
                    output = output.squeeze()
                    y_score = torch.sigmoid(output).cpu().numpy()
                    start = prev_start
                    end = start + data.shape[0]
                    y_scores[i, start:end, iter] = y_score
                    prev_start = end

        y_preds_model = y_preds.mean(axis=-1) > threshold

        if has_labels:
            TP = ((dataset.y == 1) & (y_preds_model == 1)).sum().item()
            FP = ((dataset.y == 0) & (y_preds_model == 1)).sum().item()
            FN = ((dataset.y == 1) & (y_preds_model == 0)).sum().item()
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f1 = 2 * p * r / (p + r)
            print(f'{model_checkpoint} precision: {p}')
            print(f'{model_checkpoint} recall: {r}')
            print(f'{model_checkpoint} f1: {f1}')

    # average over num_iters
    y_scores = y_scores.mean(axis=-1)

    # average over len(models)
    y_scores = y_scores.mean(axis=0)

    y_preds = y_scores > threshold
    if has_labels:
        metrics.update_accuracies(y_true=dataset.y, y_score=y_scores, threshold=threshold)

    df = pd.DataFrame({'roi-id': dataset.roi_ids, 'y_score': y_scores, 'y_pred': y_preds})

    return metrics, df