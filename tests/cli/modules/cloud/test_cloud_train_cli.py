import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import sagemaker

from deepcell.cli.modules.cloud.train import CloudKFoldTrainRunner
from deepcell.cloud.ecr import ECRUploader
from deepcell.cloud.train import KFoldTrainingJobRunner
from deepcell.testing.util import get_test_data


class TestTrainCLI:
    @classmethod
    def setup_class(cls):
        data_dir = tempfile.TemporaryDirectory()

        cls.dataset = get_test_data(write_dir=data_dir.name, exp_id='0')
        cls.data_dir = data_dir

    def teardown_class(self):
        self.data_dir.cleanup()

    @patch("boto3.session")
    @patch('docker.APIClient')
    @patch('deepcell.cloud.train.get_sagemaker_execution_role_arn',
           return_value='')
    @patch.object(KFoldTrainingJobRunner,
                  '_wait_until_training_jobs_have_finished')
    @patch.object(KFoldTrainingJobRunner, '_upload_local_data_to_s3')
    @patch.object(sagemaker.estimator.Estimator, '__init__', return_value=None)
    @patch.object(sagemaker.estimator.Estimator, 'fit')
    @patch.object(ECRUploader, '_docker_login', return_value=('', ''))
    @pytest.mark.parametrize('local_mode', [True, False])
    def test_cli(self, _, __, ___, ____, _____, ______, _______, ________,
                 local_mode):
        """Smoke tests the CLI"""
        instance_type = 'local' if local_mode else 'ml.p3.2xlarge'

        with tempfile.TemporaryDirectory() as temp_path:
            with open(Path(temp_path) / 'model_inputs.json', 'w') as f:
                json.dump(self.dataset, f)

            with open(Path(temp_path) / 'model_inputs.json', 'r') as f:
                train_params = {
                    'n_folds': 2,
                    'model_inputs_path': f.name,
                    'save_path': temp_path,
                    'optimization_params': {
                        'n_epochs': 3
                    }
                }

                input_data = {
                    'train_params': train_params,
                    'instance_type': instance_type
                }
                trainer = CloudKFoldTrainRunner(
                    input_data=input_data, args=[])
                trainer.run()
