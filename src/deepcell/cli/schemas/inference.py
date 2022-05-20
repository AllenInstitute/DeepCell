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
        required=False,
        default=None,
        allow_none=True,
        description='If provided, we are running inference on a specific '
                    'experiment'
    )
    has_labels = argschema.fields.Bool(
        default=False,
        description='Whether we have labels (test set) or do not'
    )
    model_params = argschema.fields.Nested(
        ModelSchema,
        default={}
    )
