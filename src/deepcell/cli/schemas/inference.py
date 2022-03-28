import argschema

from deepcell.cli.schemas.base import UnsplitBaseSchema
from deepcell.cli.schemas.model import ModelSchema


class InferenceSchema(UnsplitBaseSchema):
    experiment_id = argschema.fields.String(
        required=True,
        description='What experiment to run inference on'
    )
    model_params = argschema.fields.Nested(
        ModelSchema,
        default={}
    )
