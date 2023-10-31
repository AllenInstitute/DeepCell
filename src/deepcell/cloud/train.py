import contextlib
import datetime
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, Union, List, Dict, Iterator, Tuple

import boto3.session
import mlflow
import sagemaker
from sagemaker.estimator import Estimator

from deepcell.aws_utils import get_sagemaker_execution_role_arn, \
    create_bucket_if_not_exists, download_from_s3
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput, \
    copy_model_inputs_to_dir, write_model_input_metadata_to_disk
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
                 mlflow_experiment_name: str = 'deepcell-train',
                 s3_data_key: Optional[str] = None,
                 load_pretrained_checkpoints_path: Optional[Path] = None
                 ):
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
        s3_data_key: If provided, loads data from this key rather than
            uploading new data
            This is only used if not in "local mode"
        load_pretrained_checkpoints_path
            Load pretrained weights for finetuning from this path
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
        self._load_pretrained_checkpoints_path = (
            load_pretrained_checkpoints_path)
        self._logger = logging.getLogger(__name__)
        sh = logging.StreamHandler()
        sh.setLevel(level=logging.INFO)
        sh.setFormatter(fmt=logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self._logger.addHandler(sh)

        self._s3_data_key = s3_data_key
        self._input_data_root_dir = os.path.join(
            'input_data',
            s3_data_key if s3_data_key is not None else
            datetime.datetime.now().strftime('%Y-%m-%d_%H:%m:%S-%f'))

        if mlflow_server_uri is not None:
            mlflow.set_tracking_uri(mlflow_server_uri)

        boto_session = boto3.session.Session(profile_name=profile_name,
                                             region_name=region_name)
        self._sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, default_bucket=bucket_name)

    def run(self,
            train_params: dict,
            model_inputs: Optional[List[ModelInput]] = None,
            k_folds=5,
            is_trial_run=False,
            limit_val_to_projects: Optional[List[str]] = None
            ):
        """
        Train the model using `model_inputs` on sagemaker

        Parameters
        ----------
        model_inputs: the input data. If provided, data_load_path should not be
        train_params: Training parameters to log to MLFlow
        k_folds: The number of CV splits.
        is_trial_run: Set this to True to only run on a single fold. Useful
                    for trying out a new idea without running on every fold
        limit_val_to_projects: Limit the validation set to this list of
            projects

        Returns
        -------
        None
        """
        if self._local_mode:
            if model_inputs is None:
                raise ValueError('model_inputs is required in local mode')
        if model_inputs is not None and self._s3_data_key is not None:
            raise ValueError('Supply either `model_inputs` or s3_data_key, '
                             'but not both')
        if model_inputs is None and self._s3_data_key is None:
            raise ValueError('Supply one of `model_inputs` or s3_data_key')
        sagemaker_session = None if self._local_mode else \
            self._sagemaker_session

        sagemaker_role_arn = get_sagemaker_execution_role_arn()

        s3_uri = f's3://{self._bucket_name}/{self._input_data_root_dir}'
        if self._s3_data_key is not None:
            model_inputs = self._get_model_inputs_from_s3(s3_uri=s3_uri)
        else:
            if not self._local_mode:
                # If we are not running locally and we are not loading data
                # from s3, upload all model_input metadata to S3
                with tempfile.TemporaryDirectory() as temp_dir:
                    write_model_input_metadata_to_disk(
                        path=Path(temp_dir) / 'all' / 'model_inputs.json',
                        model_inputs=model_inputs
                    )
                    self._sagemaker_session.upload_data(
                        path=str(Path(temp_dir) / 'all' / 'model_inputs.json'),
                        key_prefix=f'{self._input_data_root_dir}/all',
                        bucket=self._bucket_name)

        if self._is_mlflow_tracking_enabled:
            def _get_channel_order(model_inputs: List[ModelInput]):
                channel_order = set(tuple(x.channel_order)
                                    for x in model_inputs)
                if len(channel_order) != 1:
                    raise ValueError(f'Expected to find 1 channel order but '
                                     f'found {channel_order}')
                return list(channel_order)[0]

            train_params['data_params']['channel_order'] = \
                _get_channel_order(model_inputs=model_inputs)
            run_name = (
                f'CV-{int(time.time())}'
                if train_params['tracking_params']['mlflow_run_name'] is None
                else train_params['tracking_params']['mlflow_run_name']
            )
            mlflow_run = self._create_parent_mlflow_run(
                run_name=run_name,
                hyperparameters=train_params,
                hyperparameters_exclude_keys=['tracking_params', 'save_path',
                                              'model_load_path',
                                              'model_inputs_path',
                                              'model_inputs', 'n_folds',
                                              'data_load_path',
                                              's3_data_key'])
        else:
            mlflow_run = contextlib.nullcontext()
        channels = ('train', 'validation')

        training_jobs = []

        with mlflow_run:
            for k, (train_idx, test_idx) in enumerate(
                    DataSplitter.get_cross_val_split_idxs(
                        model_inputs=model_inputs, seed=self._seed,
                        n_splits=k_folds)):
                if is_trial_run:
                    if k != 0:
                        continue
                train = [model_inputs[i] for i in train_idx]
                test = [model_inputs[i] for i in test_idx]

                output_dir = f'file://{Path(self._output_dir) / str(k)}' if \
                    self._local_mode else None
                job_name = None if self._local_mode else \
                    f'deepcell-train-fold-{k}-{int(time.time())}'
                env_vars = {
                    'fold': f'{k}',
                    'load_pretrained_checkpoints_path': (
                        str(self._load_pretrained_checkpoints_path) if
                        self._load_pretrained_checkpoints_path is not None
                        else None),
                    'mlflow_server_uri': self._mlflow_server_uri,
                    'mlflow_experiment_name': self._mlflow_experiment_name,
                    'sagemaker_job_name': job_name,
                    'mlflow_parent_run_id':
                        (mlflow_run.info.run_id if
                         self._is_mlflow_tracking_enabled else None)
                }

                # Due to
                # https://github.com/aws/sagemaker-python-sdk/issues/2930
                hyperparameters = env_vars if self._local_mode else {}

                with tempfile.TemporaryDirectory() as temp_dir:
                    local_data_path = {c: Path(temp_dir) / c / str(k)
                                       for c in channels}
                    if self._s3_data_key is None:
                        self._logger.info(f'Copying train model inputs to '
                                          f'{local_data_path["train"]}')
                        copy_model_inputs_to_dir(
                            destination=local_data_path['train'],
                            model_inputs=train
                        )
                        self._logger.info(
                            f'Copying validation model inputs '
                            f'to {local_data_path["validation"]}')
                        copy_model_inputs_to_dir(
                            destination=local_data_path['validation'],
                            model_inputs=test
                        )
                    if self._local_mode:
                        # In local mode, we are just reading local data
                        data_path = {k: f'file://{v}'
                                     for k, v in local_data_path.items()}
                    elif self._s3_data_key is None:
                        # If not reading data from S3,
                        # we need to upload data to s3
                        data_path = self._upload_local_data_to_s3(
                            k=k, local_data_path=local_data_path)
                    else:
                        # We are loading data from s3
                        data_path = {c: f'{s3_uri}/{c}/{k}' for c in channels}

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

                    estimator.fit(
                        inputs=data_path,
                        wait=False,
                        job_name=job_name
                    )
                    training_jobs.append(job_name)

                    # sleeps to avoid Sagemaker Training Job API throttling
                    time.sleep(5)
            self._wait_until_training_jobs_have_finished(
                training_jobs=training_jobs)
            self._log_cross_validation_end_metrics_to_mlflow()

    def _upload_local_data_to_s3(
            self,
            local_data_path: Dict[str, Path],
            k: int
    ) -> Dict[str, str]:
        """
        Uploads local data to s3

        Parameters
        ----------
        local_data_path: dict mapping sagemaker "channel" to Path on disk
        k: the current fold

        Returns
        -------
        Dict mapping channel name to path on s3
        """
        create_bucket_if_not_exists(
            bucket=self._bucket_name,
            region_name=self._sagemaker_session.boto_session
            .region_name)
        s3_paths = {}
        for channel in local_data_path:
            self._logger.info(f'Compressing {local_data_path[channel]}')
            compressed = shutil.make_archive(
                base_name=str(local_data_path[channel]),
                format='gztar',
                root_dir=str(local_data_path[channel])
            )
            self._logger.info(f'Done compressing {local_data_path[channel]}')

            s3_key_prefix = f'{self._input_data_root_dir}/{channel}/{k}'
            self._logger.info(f'Uploading {compressed} to '
                              f's3://{self._bucket_name}/{s3_key_prefix}')
            s3_path = self._sagemaker_session.upload_data(
                path=compressed,
                key_prefix=s3_key_prefix,
                bucket=self._bucket_name)
            s3_paths[channel] = s3_path

        return s3_paths

    @staticmethod
    def _get_model_inputs_from_s3(s3_uri: str,
                                  fold: Optional[int] = None,
                                  is_train=False) \
            -> List[ModelInput]:
        """
        Get model inputs from s3
        @param s3_uri: Path on s3 to read data from
        @param fold: Optional fold, to get model inputs for just that fold
        @param is_train: Whether to get train or validation model inputs
            (only applicable with `fold` passed)
        @return: List[ModelInput]
        """
        if fold is not None:
            train_or_validation = 'train/' if is_train else 'validation/'
        else:
            train_or_validation = ''
        with download_from_s3(
                uri=f'{s3_uri}/'
                    f'{fold or "all"}/'
                    f'{train_or_validation}model_inputs.json')['Body'] as f:
            model_inputs = json.load(f)
        model_inputs = [ModelInput(**x) for x in model_inputs]
        return model_inputs

    def _wait_until_training_jobs_have_finished(
            self, training_jobs: List[str]):
        """Blocks until all training jobs have completed

          Args:
             training_jobs: array of submitted training job names

        """
        all_jobs_done = False
        while not all_jobs_done:
            completed_jobs = 0
            for job in training_jobs:
                job_detail = self._sagemaker_session.sagemaker_client\
                    .describe_training_job(TrainingJobName=job)
                job_status = job_detail['TrainingJobStatus']
                if job_status.lower() in ('completed', 'failed', 'stopped'):
                    completed_jobs += 1
            if completed_jobs == len(training_jobs):
                all_jobs_done = True
            else:
                time.sleep(30)
