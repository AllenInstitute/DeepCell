"""
Trains the model.
"""
import json
import logging
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

from deepcell.cli.modules.train import TrainRunner
from deepcell.cli.schemas.data import ModelInputSchema
from deepcell.datasets.model_input import ModelInput, \
    write_model_input_metadata_to_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path('/opt/ml/input/data/train')
VALIDATION_DATA_DIR = Path('/opt/ml/input/data/validation')
HYPERPARAMS_PATH = '/opt/ml/input/config/hyperparameters.json'
OUTPUT_PATH = Path('/opt/ml/model')
INPUT_JSON_PATH = '/opt/ml/train_input.json'
PRETRAINED_CHECKPOINTS_PATH = '/opt/ml/model_checkpoints'


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
        self._load_pretrained_checkpoints_path = (
            self._get_input_argument(name='load_pretrained_checkpoints_path'))

        if self._fold is None:
            raise ValueError('Could not get fold from hyperparams or env.')

        for data_dir in (TRAINING_DATA_DIR, VALIDATION_DATA_DIR):
            logger.info(f'Unpacking {data_dir / self._fold}')
            shutil.unpack_archive(
                filename=f'{data_dir / self._fold}.tar.gz',
                extract_dir=data_dir
            )
            logger.info(f'Done unpacking {data_dir / self._fold}')

    def run(self):
        self._update_train_cfg()
        train = self._update_model_inputs_paths(
            data_dir=TRAINING_DATA_DIR)
        validation = self._update_model_inputs_paths(
            data_dir=VALIDATION_DATA_DIR)

        assert len(set([(x.experiment_id, x.roi_id) for x in train])
               .intersection(
            set([(x.experiment_id, x.roi_id) for x in validation]))) == 0
        print(f'Training size: {len(train)}')
        print(f'Validation size: {len(validation)}')

        write_model_input_metadata_to_disk(
            model_inputs=train,
            path=TRAINING_DATA_DIR / 'model_inputs.json')
        write_model_input_metadata_to_disk(
            model_inputs=validation,
            path=VALIDATION_DATA_DIR / 'model_inputs.json')
        sys.argv = sys.argv[:-1]
        train_runner = TrainRunner(input_data=self._train_cfg, args=[])
        train_runner.run()

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
            for channel in model_input['channel_path_map']:
                model_input['channel_path_map'][channel] = \
                    str(data_dir /
                        Path(model_input['channel_path_map'][channel]).name)

        model_inputs_serialized: List[ModelInput] = \
            ModelInputSchema().load(model_inputs, many=True)  # noqa

        return model_inputs_serialized

    def _update_train_cfg(self) -> None:
        """Update train cfg

        Returns
        --------
        None, updates train_cfg in place
        """
        self._train_cfg['save_path'] = str(OUTPUT_PATH / f'{self._fold}')
        self._train_cfg['train_model_inputs_path'] = \
            str(TRAINING_DATA_DIR / 'model_inputs.json')
        self._train_cfg['validation_model_inputs_path'] = \
            str(VALIDATION_DATA_DIR / 'model_inputs.json')
        self._train_cfg['fold'] = self._fold
        self._train_cfg['model_load_path'] = (
            self._load_pretrained_checkpoints_path)
        self._train_cfg['tracking_params']['sagemaker_job_name'] = \
            self._get_input_argument(name='sagemaker_job_name')
        self._train_cfg['tracking_params']['parent_run_id'] = \
            self._get_input_argument(name='mlflow_parent_run_id')
        self._train_cfg['tracking_params']['mlflow_server_uri'] = \
            self._get_input_argument(name='mlflow_server_uri')

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
