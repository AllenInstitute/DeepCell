"""
Trains the model.
"""
import json
import logging
import os
import argparse
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from deepcell.cli.schemas.data import ModelInputSchema
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path(os.environ['SM_CHANNEL_TRAIN'])
VALIDATION_DATA_DIR = Path(os.environ['SM_CHANNEL_VALIDATION'])

INPUT_JSON_PATH = '/opt/ml/train_input.json'


def main():
    with open(INPUT_JSON_PATH) as f:
        train_cfg = json.load(f)

    train_cfg = _update_paths_to_container_paths(train_cfg=train_cfg)

    with open(TRAINING_DATA_DIR / 'model_inputs.json') as f:
        model_inputs = json.load(f)
        train = [ModelInput(**x) for x in model_inputs]
    with open(VALIDATION_DATA_DIR / 'model_inputs.json') as f:
        model_inputs = json.load(f)
        validation = [ModelInput(**x) for x in model_inputs]

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
        model_load_path=model_params['model_load_path'],
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
        eval_fold=TRAINING_DATA_DIR.name)


def _update_paths_to_container_paths(train_cfg: Dict) -> Dict:
    """Update paths from local paths to container paths.
    The paths in the input json are expected to be local paths, but we need
    them to be container paths.


    Parameters
    ----------
    train_cfg: Train config

    Returns
    --------
    The train cfg with paths updated
    """
    for data_dir in (TRAINING_DATA_DIR, VALIDATION_DATA_DIR):
        with open(data_dir / 'model_inputs.json') as f:
            model_inputs = json.load(f)

        # Update model input paths
        for model_input in model_inputs:
            model_input['mask_path'] = (
                data_dir / Path(model_input['mask_path']).name)
            model_input['max_projection_path'] = (
                data_dir / Path(model_input['max_projection_path']).name)
            if model_input['avg_projection_path'] is not None:
                model_input['avg_projection_path'] = (
                    data_dir /
                    Path(model_input['avg_projection_path']).name)
            if model_input['correlation_projection_path'] is not None:
                model_input['correlation_projection_path'] = (
                    data_dir /
                    Path(model_input['correlation_projection_path']).name)

        with open(data_dir / 'model_inputs.json', 'w') as f:
            f.write(ModelInputSchema().dumps(model_inputs))

    # Update output path
    train_cfg['save_path'] = os.environ['SM_OUTPUT_DATA_DIR']

    return train_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train',
        help='This is a dummy argument that doesn\'t do '
        'anything. It is required due to the way '
        'sagemaker runs the container. Sagemaker always passes an argument of '
        '"train". See https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html')   # noqa E501
    args = parser.parse_args()

    main()
