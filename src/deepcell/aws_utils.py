import logging
from urllib.parse import urlparse

import boto3

logger = logging.getLogger(__name__)


def get_sagemaker_execution_role_arn() -> str:
    """
    Gets the sagemaker execution role arn
    @return: The sagemaker execution role arn

    @raise
    RuntimeError if the role cannot be found
    """
    iam = boto3.client('iam')
    roles = iam.list_roles(PathPrefix='/service-role/')
    sm_roles = [x for x in roles['Roles'] if
                x['RoleName'].startswith('AmazonSageMaker-ExecutionRole')]
    if sm_roles:
        sm_role = sm_roles[0]
    else:
        raise RuntimeError('Could not find the sagemaker execution role. '
                           'It should have already been created in AWS')
    return sm_role['Arn']


def create_bucket_if_not_exists(bucket: str,
                                region_name: str):
    """
    Creates an s3 bucket with name `bucket` if it doesn't exist
    @param region_name: region name to create bucket in
    @param bucket: bucket to create

    @return None, creates bucket
    """
    s3 = boto3.client('s3')
    buckets = s3.list_buckets()
    buckets = buckets['Buckets']
    buckets = [x for x in buckets if x['Name'] == bucket]

    if len(buckets) == 0:
        logger.info(f'Creating bucket {bucket}')
        s3.create_bucket(
            ACL='private',
            Bucket=bucket,
            CreateBucketConfiguration={
                'LocationConstraint': region_name
            }
        )


def download_from_s3(uri: str):
    s3 = boto3.client('s3')
    parsed_s3 = urlparse(uri)
    bucket = parsed_s3.netloc
    file_key = parsed_s3.path[1:]
    response = s3.get_object(Bucket=bucket, Key=file_key)
    return response
