import os.path
from pathlib import Path

import argschema
import marshmallow
from marshmallow.validate import OneOf

from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.exp_metadata import ExperimentMetadata


class ModelInputSchema(argschema.ArgSchema):
    """Defines settings for a single model input"""
    experiment_id = argschema.fields.String(
        required=True,
        description='experiment_id'
    )
    roi_id = argschema.fields.Int(
        required=True,
        description='roi id'
    )
    correlation_projection_path = argschema.fields.InputFile(
        default=None,
        allow_none=True,
        description='correlation projection path'
    )
    max_projection_path = argschema.fields.InputFile(
        required=True,
        description='max projection path'
    )
    avg_projection_path = argschema.fields.InputFile(
        default=None,
        allow_none=True,
        description='avg projection path'
    )
    mask_path = argschema.fields.InputFile(
        required=True,
        description='mask path'
    )
    label = argschema.fields.String(
        default=None,
        allow_none=True,
        description='label. Null in case of inference.',
        validate=OneOf((None, 'cell', 'not cell'))
    )

    @marshmallow.post_load
    def make_model_input(self, data):
        data = {k: v for k, v in data.items() if k not in ('log_level',)}
        for k, v in data.items():
            if isinstance(v, str):
                if os.path.isfile(v):
                    data[k] = Path(v)

        return ModelInput(**data)


class ExperimentMetadataSchema(argschema.ArgSchema):
    """Defines metadata for experiments"""
    experiment_id = argschema.fields.String(
        required=True,
        description='experiment_id'
    )
    imaging_depth = argschema.fields.Int(
        required=True,
        description='Depth of the experiment'
    )
    equipment = argschema.fields.String(
        required=True,
        description='Name of the equipment doing data acquisition.',
    )
    problem_experiment = argschema.fields.Bool(
        required=True,
        description='Marker that the experiment is special and should be '
                    'considered a unique sample.',
    )

    @marshmallow.post_load
    def make_experiment_metadata(self, data):
        data = {k: v for k, v in data.items() if k not in ('log_level',)}
        return ExperimentMetadata(**data)


class DataSchema(argschema.ArgSchema):
    """Defines settings for model inputs"""
    crop_size = argschema.fields.Tuple(
        (argschema.fields.Int, argschema.fields.Int),
        default=(128, 128),
        description='Width, height center crop size to apply to all inputs'
    )

    center_roi_centroid = argschema.fields.Bool(
        default=False,
        description='Manually center input by finding centroid of soma'
    )
