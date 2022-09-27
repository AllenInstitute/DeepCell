import warnings
from pathlib import Path
from typing import Tuple, Union, Dict, List, Optional

import numpy as np
import os
import pandas as pd

import torch
from sklearn.metrics import precision_score, recall_score, \
    average_precision_score
from torch.utils.data import DataLoader

from deepcell.datasets.model_input import ModelInput
from deepcell.metrics import Metrics
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.data_splitter import DataSplitter
from deepcell.transform import Transform


def inference(model: torch.nn.Module,
              test_loader: DataLoader,
              checkpoint_path: str,
              has_labels=True,
              threshold=0.5,
              ensemble=True,
              cv_fold=None,
              tta_num_iters=0,
              tta_stat='avg') -> Tuple[Metrics, pd.DataFrame]:
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
        tta_num_iters:
            Number of times to perform test-time augmentation (default 0)
        tta_stat
            Stat to apply to aggregate tta iters

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
        models = [model for model in models if model != 'model_init.pt' and
                  Path(model).suffix == '.pt']
        if len(models) == 0:
            raise RuntimeError('Could not find model')
    else:
        models = [f'{cv_fold}_model.pt']

    num_iters = 1 if tta_num_iters == 0 else tta_num_iters

    y_scores = np.zeros((len(models), len(dataset), num_iters))

    use_cuda = torch.cuda.is_available()
    print(f'CUDA is available: {use_cuda}')

    for i, model_checkpoint in enumerate(models):
        map_location = None if use_cuda else torch.device('cpu')
        x = torch.load(f'{checkpoint_path}/{model_checkpoint}',
                       map_location=map_location)

        if 'state_dict' in x:
            state_dict = x['state_dict']
        else:
            # backwards compatability
            state_dict = x
        model.load_state_dict(state_dict=state_dict)

        model.eval()

        if use_cuda:
            model.cuda()

        for iter in range(num_iters):
            prev_start = 0

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

        y_preds_model = y_scores[i].mean(axis=-1) > threshold

        if has_labels:
            TP = ((dataset.y == 1) & (y_preds_model == 1)).sum().item()
            FP = ((dataset.y == 0) & (y_preds_model == 1)).sum().item()
            FN = ((dataset.y == 1) & (y_preds_model == 0)).sum().item()

            if TP + FP == 0 or TP + FN == 0:
                warnings.warn('Division by zero')
            else:
                p = TP / (TP + FP)
                r = TP / (TP + FN)
                f1 = 2 * p * r / (p + r)
                print(f'{model_checkpoint} precision: {p}')
                print(f'{model_checkpoint} recall: {r}')
                print(f'{model_checkpoint} f1: {f1}')

    # average over num_iters
    if tta_num_iters > 0:
        if tta_stat == 'avg':
            y_scores = y_scores.mean(axis=-1)
        elif tta_stat == 'max':
            y_scores = y_scores.max(axis=-1)
        else:
            raise ValueError('tta_stat must by either "max" or "avg"')
    else:
        y_scores = y_scores.mean(axis=-1)

    # average over len(models)
    y_scores = y_scores.mean(axis=0)

    y_preds = y_scores > threshold
    if has_labels:
        metrics.update_accuracies(y_true=dataset.y, y_score=y_scores, threshold=threshold)

    roi_ids = [x.roi_id for x in dataset.model_inputs]
    experiment_ids = [x.experiment_id for x in dataset.model_inputs]

    df = pd.DataFrame({
        'roi-id': roi_ids,
        'experiment_id': experiment_ids,
        'y_score': y_scores,
        'y_pred': y_preds
    })

    if has_labels:
        df['y_true'] = [x.label for x in dataset.model_inputs]

    return metrics, df


def cv_performance(
        model: torch.nn.Module,
        model_inputs: Union[List[ModelInput], List[List[ModelInput]]],
        checkpoint_path: Union[str, Path],
        data_splitter: Optional[DataSplitter] = None,
        threshold=0.5,
        test_transform: Optional[Transform] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluates each of the k trained models on the respective validation set
    Returns the CV predictions as well as performance stats

    Args:
        model:
            Model to evaluate
        model_inputs:
            List of model inputs
        data_splitter
            DataSplitter, to perform train/val split
        checkpoint_path:
            Model weights to load
        threshold
            classification threshold
        test_transform
            Test transform to pass to RoiDataset (in case of no DataSplitter)
    Returns:
        Dataframe of predictions, precision, recall, aupr (mean, std) across
        folds
    """
    if data_splitter is None and type(model_inputs[0]) != list:
        raise RuntimeError('If no data splitter is passed, a list of list '
                           'of ModelInput is expected, where each list is '
                           'a single validation set')
    if data_splitter is None and test_transform is None:
        raise ValueError('Pass test_transform if no data splitter')
    if data_splitter is not None and type(model_inputs[0]) == list:
        raise RuntimeError('If a data splitter is passed, a list '
                           'of ModelInput is expected')

    def get_validation_set():
        if data_splitter is None:
            for k in range(len(model_inputs)):
                yield RoiDataset(model_inputs=model_inputs[k],
                                 transform=test_transform)
        else:
            for _, val in data_splitter.get_cross_val_split(
                    train_dataset=RoiDataset(model_inputs=model_inputs)):
                yield val

    y_scores = []
    y_preds = []
    y_true = []
    roi_ids = []
    experiment_ids = []
    precisions = []
    recalls = []
    auprs = []

    for k, val in enumerate(get_validation_set()):
        val_loader = DataLoader(dataset=val, shuffle=False, batch_size=64)
        _, res = inference(model=model, test_loader=val_loader,
                           threshold=threshold,
                           cv_fold=k, ensemble=False,
                           checkpoint_path=checkpoint_path)

        res['y_true'] = val.y

        y_scores += res['y_score'].tolist()
        y_preds += res['y_pred'].tolist()
        y_true += res['y_true'].tolist()
        experiment_ids += res['experiment_id'].tolist()
        roi_ids += res['roi-id'].tolist()

        precision = precision_score(y_pred=res['y_pred'], y_true=res['y_true'])
        recall = recall_score(y_pred=res['y_pred'], y_true=res['y_true'])
        aupr = average_precision_score(y_score=res['y_score'],
                                       y_true=res['y_true'])
        precisions.append(precision)
        recalls.append(recall)
        auprs.append(aupr)

    df = pd.DataFrame(
        {
            'experiment_id': experiment_ids,
            'roi_id': roi_ids,
            'y_true': y_true,
            'y_score': y_scores,
            'y_pred': y_preds
        })
    df['error'] = (df['y_true'] - df['y_score']).abs().round(2)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    auprs = np.array(auprs)
    metrics = {
        'precision': (precisions.mean(), precisions.std()),
        'recall': (recalls.mean(), recalls.std()),
        'aupr': (auprs.mean(), auprs.std())
    }
    return df.sort_values('error', ascending=False), metrics
