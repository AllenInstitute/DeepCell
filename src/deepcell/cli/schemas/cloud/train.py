import argschema

from deepcell.cli.schemas.train import KFoldCrossValidationSchema as \
    KFoldCrossValidationBaseSchema


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

    image_uri = argschema.fields.Str(
        default=None,
        allow_none=True,
        description='URI to a prebuilt docker container on ECR. '
                    'This URI can be obtained from ECR on AWS'
    )


class S3ParamsSchema(argschema.ArgSchema):
    bucket_name = argschema.fields.Str(
        default='prod.deepcell.alleninstitute.org',
        description='Bucket on S3 to use for storing input data, '
                    'model outputs. Will be created if it doesn\'t '
                    'exist. Note that the bucket name must be unique across '
                    'all AWS accounts'
    )
    data_key = argschema.fields.Str(
        default=None,
        allow_none=True,
        description='If provided, will pull data from this key on s3, rather'
                    'than uploading it. Should be what comes after input_data/'
    )


class KFoldCrossValidationSchema(KFoldCrossValidationBaseSchema):

    # Note: in `KFoldCrossValidationBaseSchema`, `model_inputs_path` should
    # always be required. But in this cloud schema, it is optional as
    # `s3_params.data_key` can be used instead
    model_inputs_path = argschema.fields.InputFile(
        default=None,
        allow_none=True,
        description='Path to json file for input examples where each '
                    'instance has schema given by '
                    '`deepcell.cli.schemas.data.ModelInputSchema`. '
                    'Note: if not provided, `s3_params.data_key` must be '
                    'provided.'
    )


class CloudKFoldTrainSchema(argschema.ArgSchema):
    train_params = argschema.fields.Nested(
        KFoldCrossValidationSchema,
        required=True
    )

    profile_name = argschema.fields.Str(
        default='default',
        description='AWS profile name. Useful for debugging to use a sandbox '
                    'account'
    )

    instance_type = argschema.fields.Str(
        required=True,
        default='ml.p3.2xlarge',
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

    is_trial_run = argschema.fields.Bool(
        default=False,
        description='Set this to True to only run on a single fold. Useful'
                    'for trying out a new idea without running on every fold'
    )
