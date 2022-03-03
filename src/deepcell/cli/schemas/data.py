import argschema

from argschema.schemas import DefaultSchema
class DataSchema(DefaultSchema):
    crop_size = argschema.fields.Tuple(
        (argschema.fields.Int, argschema.fields.Int),
        default=(128, 128),
        description='Width, height center crop size to apply to all inputs'
    )


class TrainDataSchema(DataSchema):
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


class InferenceDataSchema(DataSchema):
    data_dir = argschema.fields.InputDir(
        required=True,
        description='Path to model inputs'
    )
    rois_path = argschema.fields.InputFile(
        required=True,
        description='Path to rois'
    )
    center_roi_centroid = argschema.fields.Bool(
        default=False,
        description='Manually center input by finding centroid of soma'
    )

