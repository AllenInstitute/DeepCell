import argschema

from deepcell.cli.schemas.data import DataSchema
from deepcell.cli.schemas.model import InferenceModelSchema


class InferenceSchema(argschema.ArgSchema):
    experiment_id = argschema.fields.String(
        required=True,
        description='What experiment to run inference on'
    )
    data_params = argschema.fields.Nested(
        DataSchema,
        default={}
    )
    model_params = argschema.fields.Nested(
        InferenceModelSchema,
        default={}
    )
    out_dir = argschema.fields.OutputDir(
        required=True,
        description='Where to save predictions'
    )
