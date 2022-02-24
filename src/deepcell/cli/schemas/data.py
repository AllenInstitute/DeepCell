import argschema


class DataSchema(argschema.ArgSchema):
    download_path = argschema.fields.OutputDir(
        required=True,
        description='Where to download data from s3 to'
    )
    exclude_projects = argschema.fields.List(
        argschema.fields.String,
        default=[],
        description='List of project names to exclude from the dataset. '
                    'This is a relic from using AWS ground truth '
                    'where project names were used for the labeling jobs. '
                    'All data added to S3 is given a project name now.'
    )
    crop_size = argschema.fields.Tuple(
        (argschema.fields.Int, argschema.fields.Int),
        default=(128, 128),
        description='Width, height center crop size to apply to all inputs'
    )
