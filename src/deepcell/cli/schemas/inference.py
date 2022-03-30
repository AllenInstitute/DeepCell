import argschema

from deepcell.cli.schemas.base import BaseSchema
from deepcell.cli.schemas.model import ModelSchema


class InferenceSchema(BaseSchema):
    model_inputs_path = argschema.fields.InputFile(
        required=True,
        description='Path to json file for input examples where each '
                    'instance has schema given by '
                    '`deepcell.cli.schemas.data.ModelInputSchema`'
    )
    experiment_id = argschema.fields.String(
        required=True,
        description='What experiment to run inference on'
    )
    model_params = argschema.fields.Nested(
        ModelSchema,
        default={}
    )
