import logging
import os
import shutil
from pathlib import Path

import argschema

from deepcell.cli.schemas.cloud.train import CloudKFoldTrainSchema
from deepcell.cli.schemas.train import TrainSchema
from deepcell.cloud.ecr import ECRUploader
from deepcell.cloud.train import KFoldTrainingJobRunner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class CloudKFoldTrainRunner(argschema.ArgSchemaParser):
    default_schema = CloudKFoldTrainSchema

    _container_path = \
        Path(__file__).parent.parent.parent.parent / 'cloud' / 'container'

    def run(self):
        repository_name = self.args['docker_params']['repository_name']
        image_tag = self.args['docker_params']['image_tag']

        # Write train json to container path
        with open(self._container_path / 'train_input.json', 'w') as f:
            f.write(TrainSchema().dumps(self.args['train_params']))

        os.makedirs(self._container_path / 'checkpoints')
        if self.args['model_load_path'] is not None:
            # Copy checkpoints to container context
            for file in Path(self.args['model_load_path']).iterdir():
                shutil.copy(file, self._container_path / 'checkpoints')

        if self.args['docker_params']['image_uri'] is None:
            ecr_uploader = ECRUploader(
                repository_name=repository_name,
                image_tag=image_tag,
                profile_name=self.args['profile_name']
            )
            ecr_uploader.build_and_push_container(
                dockerfile_path=self._container_path / 'Dockerfile'
            )
            image_uri = ecr_uploader.image_uri
        else:
            image_uri = self.args['docker_params']['image_uri']

        tracking_params = self.args['train_params']['tracking_params']
        runner = KFoldTrainingJobRunner(
            bucket_name=self.args['s3_params']['bucket_name'],
            s3_data_key=self.args['s3_params']['data_key'],
            image_uri=image_uri,
            profile_name=self.args['profile_name'],
            instance_type=self.args['instance_type'],
            instance_count=self.args['instance_count'],
            timeout=self.args['timeout'],
            volume_size=self.args['volume_size'],
            output_dir=self.args['train_params']['save_path'],
            mlflow_server_uri=tracking_params['mlflow_server_uri'],
            mlflow_experiment_name=tracking_params['mlflow_experiment_name'],
            seed=1234,
            load_pretrained_checkpoints_path=self.args['model_load_path']
        )
        runner.run(
            model_inputs=self.args['train_params']['model_inputs'],
            k_folds=self.args['train_params']['n_folds'],
            train_params=self.args['train_params'],
            is_trial_run=self.args['is_trial_run']
        )


if __name__ == "__main__":
    train_cli = CloudKFoldTrainRunner()
    train_cli.run()
