import argschema

from deepcell.cli.schemas.train import TrainSchema


class DockerSchema(argschema.ArgSchema):
    repository_name = argschema.fields.Str(
        default='train-deepcell',
        description='Docker repository name that will be created, and also '
                    'the base name for the sagemaker training job (training '
                    'job will be <repository name>-<timestamp>)'
    )

    image_tag = argschema.fields.Str(
        default='latest',
        description='Image tag'
    )


class S3ParamsSchema(argschema.ArgSchema):
    bucket_name = argschema.fields.Str(
        default='prod.deepcell.alleninstitute.org',
        description='Bucket on S3 to use for storing input data, '
                    'model outputs. Will be created if it doesn\'t '
                    'exist. Note that the bucket name must be unique across '
                    'all AWS accounts'
    )


class CloudTrainSchema(argschema.ArgSchema):
    train_params = argschema.fields.Nested(
        TrainSchema,
        required=True
    )

    profile_name = argschema.fields.Str(
        default='default',
        description='AWS profile name. Useful for debugging to use a sandbox '
                    'account'
    )

    instance_type = argschema.fields.Str(
        required=True,
        description='EC2 instance type. For local mode '
                    '(train locally, useful for debugging), set to "local"'
    )

    instance_count = argschema.fields.Int(
        default=1,
        description='Number of EC2 instances to use'
    )

    docker_params = argschema.fields.Nested(
        DockerSchema,
        default={}
    )

    s3_params = argschema.fields.Nested(
        S3ParamsSchema,
        default={}
    )

    volume_size = argschema.fields.Int(
        default=60,
        description='Volume size to allocate in GB'
    )

    timeout = argschema.fields.Int(
        default=6 * 60 * 60,
        description='Training job timeout in seconds'
    )
