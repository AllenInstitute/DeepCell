import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Union, List, Dict

import boto3.session
import mlflow
import sagemaker
from sagemaker.estimator import Estimator

from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput, write_model_inputs_to_disk
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.tracking.mlflow_utils import MLFlowTrackableMixin


class KFoldTrainingJobRunner(MLFlowTrackableMixin):
    """
    A wrapper on sagemaker Estimator. Starts a training job using the docker
    image given by image_uri
    """
    def __init__(self,
                 image_uri: str,
                 bucket_name: str,
                 profile_name='default',
                 region_name='us-west-2',
                 instance_type: Optional[str] = None,
                 instance_count=1,
                 timeout=24 * 60 * 60,
                 volume_size=30,
                 output_dir: Optional[Union[Path, str]] = None,
                 seed=None,
                 mlflow_server_uri=None,
                 mlflow_experiment_name: str = 'deepcell-train'):
        """
        Parameters
        ----------
        image_uri
            The container image to run
        bucket_name
            The bucket to upload data to
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
        mlflow_server_uri
            MLFlow server URI. If provided, will log to MLFlow during training
        mlflow_experiment_name
            MLFlow experiment name to create runs under.
            Only used if mlflow_server_uri provided.
        """
        super().__init__(server_uri=mlflow_server_uri,
                         experiment_name=mlflow_experiment_name)
        local_mode = instance_type == 'local'
        if local_mode and output_dir is None:
            raise ValueError('Must provide output_dir if in local mode')

        self._image_uri = image_uri
        self._local_mode = local_mode
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._profile_name = profile_name
        self._bucket_name = bucket_name
        self._timeout = timeout
        self._volume_size = volume_size
        self._output_dir = output_dir
        self._seed = seed
        self._logger = logging.getLogger(__name__)

        if mlflow_server_uri is not None:
            mlflow.set_tracking_uri(mlflow_server_uri)

        boto_session = boto3.session.Session(profile_name=profile_name,
                                             region_name=region_name)
        self._sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, default_bucket=bucket_name)

    def run(self,
            model_inputs: List[ModelInput],
            train_params: dict,
            k_folds=5):
        """
        Train the model using `model_inputs` on sagemaker

        Parameters
        ----------
        model_inputs: the input data
        train_params: Training parameters to log to MLFlow
        k_folds: The number of CV splits

        Returns
        -------
        None
        """
        sagemaker_session = None if self._local_mode else \
            self._sagemaker_session
        sagemaker_role_arn = self._get_sagemaker_execution_role_arn()

        if self._is_mlflow_tracking_enabled:
            self._create_parent_mlflow_run(
                run_name=f'CV-{int(time.time())}',
                hyperparameters=train_params,
                hyperparameters_exclude_keys=['tracking_params', 'save_path',
                                              'model_load_path',
                                              'model_inputs_path',
                                              'model_inputs', 'n_folds'])

        y = RoiDataset.get_numeric_labels(model_inputs=model_inputs)
        for k, (train_idx, test_idx) in enumerate(
                DataSplitter.get_cross_val_split_idxs(
                    n=len(model_inputs), y=y, seed=self._seed,
                    n_splits=k_folds)):
            train = [model_inputs[i] for i in train_idx]
            validation = [model_inputs[i] for i in test_idx]

            output_dir = f'file://{Path(self._output_dir) / str(k)}' if \
                self._local_mode else None

            with tempfile.TemporaryDirectory() as temp_dir:
                data_path = self._prepare_data(
                    destination_dir=temp_dir, k=k, train=train,
                    test=validation)
                job_name = f'deepcell-train-fold-{k}-{int(time.time())}'
                env_vars = {
                    'fold': f'{k}',
                    'mlflow_server_uri': self._mlflow_server_uri,
                    'mlflow_experiment_name': self._mlflow_experiment_name,
                    'sagemaker_job_name': job_name
                }
                # In local mode, due to a bug, environment vars. are not
                # passed. Pass through hyperparameters instead
                hyperparameters = env_vars if self._local_mode else {}

                estimator = Estimator(
                    sagemaker_session=sagemaker_session,
                    role=sagemaker_role_arn,
                    instance_count=self._instance_count,
                    instance_type=self._instance_type,
                    image_uri=self._image_uri,
                    output_path=output_dir,
                    hyperparameters=hyperparameters,
                    volume_size=self._volume_size,
                    max_run=self._timeout,
                    environment=env_vars
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
                    wait=True,
                    job_name=job_name
                )
        self._log_cross_validation_end_metrics_to_mlflow()
        self._end_mlflow_run()

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
        write_model_inputs_to_disk(model_inputs=train,
                                   path=train_path / 'model_inputs.json')
        write_model_inputs_to_disk(model_inputs=test,
                                   path=test_path / 'model_inputs.json')

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
