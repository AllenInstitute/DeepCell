import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import sagemaker

from deepcell.cli.modules.cloud.train import CloudKFoldTrainRunner
from deepcell.cloud.ecr import ECRUploader
from deepcell.cloud.train import KFoldTrainingJobRunner
from tests.util.util import get_test_data


class TestTrainCLI:
    @classmethod
    def setup_class(cls):
        data_dir = tempfile.TemporaryDirectory()

        cls.dataset = get_test_data(write_dir=data_dir.name)
        cls.data_dir = data_dir

    def teardown_class(self):
        self.data_dir.cleanup()

    @patch("boto3.session")
    @patch('docker.APIClient')
    @patch.object(KFoldTrainingJobRunner, '_get_sagemaker_execution_role_arn',
                  return_value='')
    @patch.object(sagemaker.estimator.Estimator, '__init__', return_value=None)
    @patch.object(sagemaker.estimator.Estimator, 'fit')
    @patch.object(ECRUploader, '_docker_login', return_value=('', ''))
    @pytest.mark.parametrize('local_mode', [True, False])
    def test_cli(self, _, __, ___, ____, _____, ______, local_mode):
        """Smoke tests the CLI"""
        if local_mode:
            instance_type = None
        else:
            instance_type = 'mock_instance_type'

        with tempfile.TemporaryDirectory() as temp_path:
            with open(Path(temp_path) / 'model_inputs.json', 'w') as f:
                json.dump(self.dataset, f)

            with open(Path(temp_path) / 'model_inputs.json', 'r') as f:
                train_params = {
                    'data_params': {
                        'model_inputs_path': f.name
                    },
                    'save_path': temp_path,
                    'optimization_params': {
                        'n_epochs': 3
                    },
                    'test_fraction': 0.5,
                    'n_folds': 2
                }

                input_data = {
                    'local_mode': local_mode,
                    'train_params': train_params,
                    'instance_type': instance_type
                }
                trainer = CloudKFoldTrainRunner(
                    input_data=input_data, args=[])
                trainer.run()
