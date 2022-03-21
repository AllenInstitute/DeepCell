"""
Trains the model.
"""
import json
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List

from torch.utils.data import DataLoader

from deepcell.cli.schemas.data import ModelInputSchema
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path('/opt/ml/input/data/train')
VALIDATION_DATA_DIR = Path('/opt/ml/input/data/validation')
OUTPUT_PATH = Path('/opt/ml/model')
INPUT_JSON_PATH = '/opt/ml/train_input.json'


def main():
    with open(INPUT_JSON_PATH) as f:
        train_cfg = json.load(f)
    train_cfg = _update_train_cfg_paths(train_cfg=train_cfg)
    train = _update_model_inputs_paths(data_dir=TRAINING_DATA_DIR)
    validation = _update_model_inputs_paths(data_dir=VALIDATION_DATA_DIR)

    train_transform = RoiDataset.get_default_transforms(
        crop_size=train_cfg['data_params']['crop_size'], is_train=True)
    test_transform = RoiDataset.get_default_transforms(
        crop_size=train_cfg['data_params']['crop_size'], is_train=False)

    train = RoiDataset(
        model_inputs=train,
        transform=train_transform
    )
    validation = RoiDataset(
        model_inputs=validation,
        transform=test_transform
    )

    optimization_params = train_cfg['optimization_params']
    model_params = train_cfg['model_params']
    trainer = Trainer.from_model(
        model_architecture=model_params['model_architecture'],
        use_pretrained_model=model_params['use_pretrained_model'],
        classifier_cfg=model_params['classifier_cfg'],
        save_path=train_cfg['save_path'],
        n_epochs=optimization_params['n_epochs'],
        early_stopping_params=optimization_params['early_stopping_params'],
        dropout_prob=model_params['dropout_prob'],
        truncate_to_layer=model_params['truncate_to_layer'],
        freeze_to_layer=model_params['freeze_to_layer'],
        model_load_path=train_cfg['model_load_path'],
        learning_rate=optimization_params['learning_rate'],
        weight_decay=optimization_params['weight_decay'],
        learning_rate_scheduler=optimization_params['scheduler_params'],
    )

    train_loader = DataLoader(dataset=train, shuffle=True,
                              batch_size=train_cfg['batch_size'])
    valid_loader = DataLoader(dataset=validation, shuffle=False,
                              batch_size=train_cfg['batch_size'])

    trainer.train(
        train_loader=train_loader, valid_loader=valid_loader,
        eval_fold=_get_fold())


def _update_model_inputs_paths(
        data_dir: Path) -> List[ModelInput]:
    """Update paths from local paths to container paths.

    Parameters
    ----------
    data_dir: Path to model inputs

    Returns
    --------
    The List[ModelInput] with paths updated
    """
    with open(data_dir / 'model_inputs.json') as f:
        model_inputs = json.load(f)

    # Update model input paths
    for model_input in model_inputs:
        model_input['mask_path'] = (
            str(data_dir / Path(model_input['mask_path']).name))
        model_input['max_projection_path'] = (
            str(data_dir / Path(model_input['max_projection_path']).name))
        if model_input['avg_projection_path'] is not None:
            model_input['avg_projection_path'] = (
                str(data_dir /
                    Path(model_input['avg_projection_path']).name))
        if model_input['correlation_projection_path'] is not None:
            model_input['correlation_projection_path'] = (
                str(data_dir /
                    Path(model_input['correlation_projection_path']).name))
    model_inputs_serialized: List[ModelInput] = \
        ModelInputSchema().load(model_inputs, many=True)    # noqa

    return model_inputs_serialized


def _update_train_cfg_paths(train_cfg: Dict) -> Dict:
    """Update paths from local paths to container paths.

    Parameters
    ----------
    train_cfg: Train config

    Returns
    --------
    The train config with paths updated
    """
    train_cfg['save_path'] = OUTPUT_PATH / f'{_get_fold()}'
    return train_cfg


def _get_fold() -> int:
    """
    1. Check hyperparameters
    2. If not there, check env. var

    Returns
    -------
    fold
    """
    with open('/opt/ml/input/config/hyperparameters.json') as f:
        hyperparams = json.load(f)
    if 'fold' in hyperparams:
        fold = hyperparams['fold']
    elif os.environ.get('fold', None) is not None:
        fold = os.environ['fold']
    else:
        raise RuntimeError('Could not find fold')
    return fold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train',
        help='This is a dummy argument that doesn\'t do '
             'anything. It is required due to the way '
             'sagemaker runs the container. Sagemaker always passes an '
             'argument of '
             '"train". See https://docs.aws.amazon.com/sagemaker/latest/dg'
             '/your-algorithms-training-algo-dockerfile.html')  # noqa E501
    args = parser.parse_args()

    main()
