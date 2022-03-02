import logging
from pathlib import Path
from typing import Optional, Union

import boto3.session
import sagemaker
from sagemaker.estimator import Estimator


class TrainingJobRunner:
    """
    A wrapper on sagemaker Estimator. Starts a training job using the docker
    image given by image_uri
    """
    def __init__(self,
                 image_uri: str,
                 bucket_name: str,
                 profile_name='default',
                 region_name='us-west-2',
                 local_mode=False,
                 instance_type: Optional[str] = None,
                 instance_count=1,
                 timeout=24 * 60 * 60,
                 volume_size=30):
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
        local_mode
            Whether running locally.
        instance_type
            Instance type to use
        instance_count
            Instance count to use
        timeout
            Training job timeout in seconds
        volume_size
            Volume size to allocate in GB
        """
        if not local_mode:
            if instance_type is None:
                raise ValueError('Must provide instance type if not using '
                                 'local mode')

        self._image_uri = image_uri
        self._local_mode = local_mode
        self._instance_type = instance_type
        self._instance_count = instance_count
        self._profile_name = profile_name
        self._bucket_name = bucket_name
        self._timeout = timeout
        self._volume_size = volume_size
        self._logger = logging.getLogger(__name__)

        boto_session = boto3.session.Session(profile_name=profile_name,
                                             region_name=region_name)
        self._sagemaker_session = sagemaker.session.Session(
            boto_session=boto_session, default_bucket=bucket_name)

    def run(self, data_dir: Union[Path, str],
            output_dir: Optional[Union[Path, str]] = None):
        """
        Train the model using sagemaker

        Parameters
        ----------
        data_dir
            Directory containing training data
        output_dir
            Where to write output. Only used in local mode

        Returns
        -------
        None
        """
        if self._local_mode and output_dir is None:
            raise ValueError('Must provide output_dir if in local mode')
        output_dir = f'file://{output_dir}' if self._local_mode else None

        instance_type = 'local' if self._local_mode else self._instance_type
        sagemaker_session = None if self._local_mode else \
            self._sagemaker_session
        sagemaker_role_arn = self._get_sagemaker_execution_role_arn()

        estimator = Estimator(
            sagemaker_session=sagemaker_session,
            role=sagemaker_role_arn,
            instance_count=self._instance_count,
            instance_type=instance_type,
            image_uri=self._image_uri,
            output_path=output_dir,
            hyperparameters={},
            volume_size=self._volume_size,
            max_run=self._timeout
        )
        if self._local_mode:
            data_path = f'file://{data_dir}'
        else:
            self._logger.info('Uploading input data to S3')
            self._create_bucket_if_not_exists()
            data_path = self._sagemaker_session.upload_data(
                path=str(data_dir),
                key_prefix='input_data',
                bucket=self._bucket_name)
        estimator.fit(data_path)

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
