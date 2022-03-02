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

        # Write model inputs to container path
        shutil.copy(
            self.args['train_params']['data_params']['model_inputs_path'],
            self._container_path / 'model_inputs.json')

        ecr_uploader = ECRUploader(
            repository_name=repository_name,
            image_tag=image_tag,
            profile_name=self.args['profile_name']
        )
        ecr_uploader.build_and_push_container(
            entrypoint_script_path=self._container_path / 'train.py',
            dockerfile_path=self._container_path / 'Dockerfile'
        )

        data_dir = self._get_data_dir()

        runner = TrainingJobRunner(
            bucket_name=self.args['s3_params']['bucket_name'],
            image_uri=ecr_uploader.image_uri,
            profile_name=self.args['profile_name'],
            local_mode=self.args['local_mode'],
            instance_type=self.args['instance_type'],
            instance_count=self.args['instance_count'],
            timeout=self.args['timeout'],
            volume_size=self.args['volume_size']
        )
        runner.run(
            data_dir=data_dir,
            output_dir=self.args['train_params']['save_path'])

    def _get_data_dir(self):
        inputs = self.args['train_params']['data_params']['model_inputs']
        data_dirs = set()
        for input in inputs:
            if input.correlation_projection_path is not None:
                data_dirs.add(input.correlation_projection_path.parent)
            data_dirs.add(input.mask_path.parent)
            data_dirs.add(input.max_projection_path.parent)
            if input.avg_projection_path is not None:
                data_dirs.add(input.avg_projection_path.parent)
        if len(data_dirs) > 1:
            raise RuntimeError('Please place all data in the same directory')
        return list(data_dirs)[0]


if __name__ == "__main__":
    train_cli = CloudTrainer()
    train_cli.run()
