import logging
import shutil
from pathlib import Path

import argschema

from deepcell.cli.schemas.cloud.train import CloudTrainSchema
from deepcell.cli.schemas.train import TrainSchema
from deepcell.cloud.ecr import ECRUploader
from deepcell.cloud.train import TrainingJobRunner


class CloudTrainer(argschema.ArgSchemaParser):
    default_schema = CloudTrainSchema
    _logger = logging.getLogger(__name__)
    _container_path = \
        Path(__file__).parent.parent.parent.parent / 'cloud' / 'container'

    def run(self):
        repository_name = self.args['docker_params']['repository_name']
        image_tag = self.args['docker_params']['image_tag']

        # Write train json to container path
        with open(self._container_path / 'train_input.json', 'w') as f:
            f.write(TrainSchema().dumps(self.args['train_params']))

        ecr_uploader = ECRUploader(
            repository_name=repository_name,
            image_tag=image_tag,
            profile_name=self.args['profile_name']
        )
        ecr_uploader.build_and_push_container(
            entrypoint_script_path=self._container_path / 'train.py',
            dockerfile_path=self._container_path / 'Dockerfile'
        )

        hyperparams = self._construct_hyperparameters()
        runner = TrainingJobRunner(
            bucket_name=self.args['s3_params']['bucket_name'],
            image_uri=ecr_uploader.image_uri,
            profile_name=self.args['profile_name'],
            local_mode=self.args['local_mode'],
            instance_type=self.args['instance_type'],
            instance_count=self.args['instance_count'],
            timeout=self.args['timeout'],
            volume_size=self.args['volume_size'],
            hyperparameters=hyperparams,
            output_dir=self.args['save_path']
        )
        runner.run(
            model_inputs=self.args['train_params']['data_params']
            ['model_inputs'])

    def _construct_hyperparameters(self) -> dict:
        """Takes the hyperparameters given by the CLI and cleans them up for
        storage on sagemaker"""
        hyperparams = self.args['train_params'].copy()
        del hyperparams['log_level']
        hyperparams['data_params'] = {
            k: v for k, v in hyperparams['data_params'].items()
            if k not in ('model_inputs_path', 'log_level', 'model_inputs')}
        if 'scheduler_params' in hyperparams['optimization_params']:
            hyperparams['optimization_params']['scheduler_params'] = {
                k: v for k, v in
                hyperparams['optimization_params']['scheduler_params'].items()
                if k not in ('log_level',)
            }
        hyperparams['optimization_params']['early_stopping_params'] = {
            k: v for k, v in
            hyperparams['optimization_params']['early_stopping_params'].items()
            if k not in ('log_level',)
        }
        hyperparams['optimization_params'] = {
            k: v for k, v in hyperparams['optimization_params'].items() if
            k not in ('log_level',)
        }
        del hyperparams['save_path']
        del hyperparams['test_fraction']
        del hyperparams['n_folds']
        return hyperparams


if __name__ == "__main__":
    train_cli = CloudTrainer()
    train_cli.run()
