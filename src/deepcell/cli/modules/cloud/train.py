import logging
from pathlib import Path

import argschema

from deepcell.cli.schemas.cloud.train import CloudKFoldTrainSchema
from deepcell.cli.schemas.train import TrainSchema
from deepcell.cloud.ecr import ECRUploader
from deepcell.cloud.train import KFoldTrainingJobRunner


class CloudKFoldTrainRunner(argschema.ArgSchemaParser):
    default_schema = CloudKFoldTrainSchema
    _logger = logging.getLogger(__name__)
    _container_path = \
        Path(__file__).parent.parent.parent.parent / 'cloud' / 'container'

    def run(self):
        repository_name = self.args['docker_params']['repository_name']
        image_tag = self.args['docker_params']['image_tag']

        # Write train json to container path
        with open(self._container_path / 'train_input.json', 'w') as f:
            f.write(TrainSchema().dumps(self.args['train_params']))

        # TODO add ability to load image from ECR rather than building and
        #  uploading another one
        ecr_uploader = ECRUploader(
            repository_name=repository_name,
            image_tag=image_tag,
            profile_name=self.args['profile_name']
        )
        ecr_uploader.build_and_push_container(
            dockerfile_path=self._container_path / 'Dockerfile'
        )

        tracking_params = self.args['train_params']['tracking_params']
        runner = KFoldTrainingJobRunner(
            bucket_name=self.args['s3_params']['bucket_name'],
            image_uri=ecr_uploader.image_uri,
            profile_name=self.args['profile_name'],
            instance_type=self.args['instance_type'],
            instance_count=self.args['instance_count'],
            timeout=self.args['timeout'],
            volume_size=self.args['volume_size'],
            output_dir=self.args['train_params']['save_path'],
            mlflow_server_uri=tracking_params['mlflow_server_uri'],
            mlflow_experiment_name=tracking_params['mlflow_experiment_name']
        )
        runner.run(
            model_inputs=self.args['train_params']['model_inputs'],
            load_data_from_s3=self.args['train_params']['load_data_from_s3'],
            k_folds=self.args['train_params']['n_folds'],
            train_params=self.args['train_params']
        )


if __name__ == "__main__":
    train_cli = CloudKFoldTrainRunner()
    train_cli.run()
