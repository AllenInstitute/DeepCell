"""
Trains the model.
"""
import json
import logging
import argparse
import os
from pathlib import Path
from typing import List

from torch.utils.data import DataLoader

from deepcell.cli.schemas.data import ModelInputSchema
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path('/opt/ml/input/data/train')
VALIDATION_DATA_DIR = Path('/opt/ml/input/data/validation')
HYPERPARAMS_PATH = '/opt/ml/input/config/hyperparameters.json'
OUTPUT_PATH = Path('/opt/ml/model')
INPUT_JSON_PATH = '/opt/ml/train_input.json'


class TrainingRunner:
    """Training Runner"""
    def __init__(self):
        with open(HYPERPARAMS_PATH) as f:
            hyperparams = json.load(f)

        with open(INPUT_JSON_PATH) as f:
            train_cfg = json.load(f)

        self._train_cfg = train_cfg
        self._hyperparams = hyperparams
        self._fold = self._get_input_argument(name='fold')

        if self._fold is None:
            raise ValueError('Could not get fold from hyperparams or env.')

    def run(self):
        self._update_train_cfg_paths()
        train = self._update_model_inputs_paths(
            data_dir=TRAINING_DATA_DIR)
        validation = self._update_model_inputs_paths(
            data_dir=VALIDATION_DATA_DIR)

        train_transform = RoiDataset.get_default_transforms(
            crop_size=self._train_cfg['data_params']['crop_size'],
            is_train=True)
        test_transform = RoiDataset.get_default_transforms(
            crop_size=self._train_cfg['data_params']['crop_size'],
            is_train=False)

        train = RoiDataset(
            model_inputs=train,
            transform=train_transform
        )
        validation = RoiDataset(
            model_inputs=validation,
            transform=test_transform
        )

        optimization_params = self._train_cfg['optimization_params']
        model_params = self._train_cfg['model_params']
        trainer = Trainer.from_model(
            model_architecture=model_params['model_architecture'],
            use_pretrained_model=model_params['use_pretrained_model'],
            classifier_cfg=model_params['classifier_cfg'],
            save_path=self._train_cfg['save_path'],
            n_epochs=optimization_params['n_epochs'],
            early_stopping_params=optimization_params['early_stopping_params'],
            dropout_prob=model_params['dropout_prob'],
            truncate_to_layer=model_params['truncate_to_layer'],
            freeze_to_layer=model_params['freeze_to_layer'],
            model_load_path=self._train_cfg['model_load_path'],
            learning_rate=optimization_params['learning_rate'],
            weight_decay=optimization_params['weight_decay'],
            learning_rate_scheduler=optimization_params['scheduler_params'],
            mlflow_server_uri=self._get_input_argument(
                name='mlflow_server_uri'),
            mlflow_experiment_name=self._get_input_argument(
                name='mlflow_experiment_name')
        )

        train_loader = DataLoader(dataset=train, shuffle=True,
                                  batch_size=self._train_cfg['batch_size'])
        valid_loader = DataLoader(dataset=validation, shuffle=False,
                                  batch_size=self._train_cfg['batch_size'])

        trainer.train(
            train_loader=train_loader, valid_loader=valid_loader,
            eval_fold=self._fold)

    @staticmethod
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

    def _update_train_cfg_paths(self) -> None:
        """Update paths from local paths to container paths.

        Returns
        --------
        None, updates train_cfg in place
        """
        self._train_cfg['save_path'] = OUTPUT_PATH / f'{self._fold}'

    def _get_input_argument(self, name):
        """
        Checks for an argument
        1. in an environment variable or
        2. due to bug
        https://github.com/aws/sagemaker-python-sdk/issues/2930 in local mode,
        in hyperparameters json
        """
        return os.environ.get(name, None) or self._hyperparams.get(name, None)


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

    runner = TrainingRunner()
    runner.run()
