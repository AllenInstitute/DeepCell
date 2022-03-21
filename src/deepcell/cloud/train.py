import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Dict

import boto3.session
import sagemaker
from sagemaker.estimator import Estimator

from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset


class TrainingJobRunner:
    """
    A wrapper on sagemaker Estimator. Starts a training job using the docker
    image given by image_uri
    """
    def __init__(self,
                 image_uri: str,
                 bucket_name: str,
                 hyperparameters: dict,
                 profile_name='default',
                 region_name='us-west-2',
                 instance_type: Optional[str] = None,
                 instance_count=1,
                 timeout=24 * 60 * 60,
                 volume_size=30,
                 output_dir: Optional[Union[Path, str]] = None,
                 seed=None):
        """
        Parameters
        ----------
        image_uri
            The container image to run
        bucket_name
            The bucket to upload data to
        hyperparameters
            Hyperparameters to track in sagemaker
        profile_name
            AWS profile name to use
        region_name
            AWS region to use
        instance_type
            Instance type to use
        instance_count
            Instance count to use
        timeout
            Training job timeout in seconds
        volume_size
            Volume size to allocate in GB
        output_dir
            Where to write output. Only used in local mode
        seed
            Seed to control reproducibility
        """
        local_mode = instance_type == 'local'
        if local_mode and output_dir is None:
            raise ValueError('Must provide output_dir if in local mode')

        self._image_uri = image_uri
        self._local_mode = local_mode
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._profile_name = profile_name
        self._bucket_name = bucket_name
        self._hyperparameters = hyperparameters
        self._timeout = timeout
        self._volume_size = volume_size
        self._output_dir = output_dir
        self._seed = seed
        self._logger = logging.getLogger(__name__)

        boto_session = boto3.session.Session(profile_name=profile_name,
                                             region_name=region_name)
        self._sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, default_bucket=bucket_name)

    def run(self, model_inputs: List[ModelInput]):
        """
        Train the model using `model_inputs` on sagemaker

        Parameters
        ----------
        model_inputs: the input data

        Returns
        -------
        None
        """
        sagemaker_session = None if self._local_mode else \
            self._sagemaker_session
        sagemaker_role_arn = self._get_sagemaker_execution_role_arn()

        # TODO add train/test split before cv split

        y = RoiDataset.get_numeric_labels(model_inputs=model_inputs)
        for k, (train_idx, test_idx) in enumerate(
                DataSplitter.get_cross_val_split_idxs(
                    n=len(model_inputs), y=y, seed=self._seed)):
            train = [model_inputs[i] for i in train_idx]
            validation = [model_inputs[i] for i in test_idx]

            output_dir = f'file://{Path(self._output_dir) / str(k)}' if \
                self._local_mode else None

            with tempfile.TemporaryDirectory() as temp_dir:
                data_path = self._prepare_data(
                    destination_dir=temp_dir, k=k, train=train,
                    test=validation)

                if self._local_mode:
                    # In local mode, due to a bug, environment vars. are not
                    # passed. Pass through hyperparameters instead
                    self._hyperparameters['fold'] = k
                estimator = Estimator(
                    sagemaker_session=sagemaker_session,
                    role=sagemaker_role_arn,
                    instance_count=self._instance_count,
                    instance_type=self._instance_type,
                    image_uri=self._image_uri,
                    output_path=output_dir,
                    hyperparameters=self._hyperparameters,
                    volume_size=self._volume_size,
                    max_run=self._timeout,
                    environment={
                        'fold': f'{k}'
                    }
                )

                if not self._local_mode:
                    # TODO check if data exists on s3 or overwrite is true.
                    #  If not then upload it.
                    self._logger.info('Uploading input data to S3')
                    self._create_bucket_if_not_exists()
                    for channel in data_path:
                        s3_path = self._sagemaker_session.upload_data(
                            path=str(data_path[channel]),
                            key_prefix=f'input_data/{channel}/{k}',
                            bucket=self._bucket_name)
                        data_path[channel] = s3_path
                estimator.fit(
                    inputs=data_path,
                    # TODO change this to false to enable parallel training
                    wait=True)

    def _prepare_data(
            self,
            destination_dir: str,
            k: int,
            train: List[ModelInput],
            test: List[ModelInput],
    ) -> Dict[str, Union[str, Path]]:
        """
        Prepares input data for training

        Parameters
        ----------
        destination_dir: where to copy model inputs
        k: the current fold
        train: train dataset
        test: test dataset

        Returns
        -------
        """
        train_path = Path(destination_dir) / 'train' / f'{k}'
        test_path = Path(destination_dir) / 'validation' / f'{k}'
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Copy model inputs
        for model_input in train:
            model_input.copy(destination=train_path)
        for model_input in test:
            model_input.copy(destination=test_path)

        # Copy metadata
        with open(train_path / 'model_inputs.json', 'w') as f:
            f.write(json.dumps([x.to_dict() for x in train], indent=2))
        with open(test_path / 'model_inputs.json', 'w') as f:
            f.write(json.dumps([x.to_dict() for x in test], indent=2))

        if self._local_mode:
            train_path = f'file://{train_path}'
            test_path = f'file://{test_path}'

        return {
            'train': train_path,
            'validation': test_path
        }

    def _get_sagemaker_execution_role_arn(self) -> str:
        """
        Gets the sagemaker execution role arn
        Returns
        -------
        The sagemaker execution role arn
        Raises
        -------
        RuntimeError if the role cannot be found
        """
        iam = self._sagemaker_session.boto_session.client('iam')
        roles = iam.list_roles(PathPrefix='/service-role/')
        sm_roles = [x for x in roles['Roles'] if
                    x['RoleName'].startswith('AmazonSageMaker-ExecutionRole')]
        if sm_roles:
            sm_role = sm_roles[0]
        else:
            raise RuntimeError('Could not find the sagemaker execution role. '
                               'It should have already been created in AWS')
        return sm_role['Arn']

    def _create_bucket_if_not_exists(self):
        """
        Creates an s3 bucket with name self._bucket_name if it doesn't exist
        Returns
        -------
        None, creates bucket
        """
        s3 = self._sagemaker_session.boto_session.client('s3')
        buckets = s3.list_buckets()
        buckets = buckets['Buckets']
        buckets = [x for x in buckets if x['Name'] == self._bucket_name]

        if len(buckets) == 0:
            self._logger.info(f'Creating bucket {self._bucket_name}')
            region_name = self._sagemaker_session.boto_session.region_name
            s3.create_bucket(
                ACL='private',
                Bucket=self._bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': region_name
                }
            )
