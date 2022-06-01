import argschema
from marshmallow.validate import OneOf

from deepcell.cli.schemas.base import BaseSchema
from deepcell.cli.schemas.model import ModelSchema


class InferenceSchema(BaseSchema):
    model_inputs_paths = argschema.fields.List(
        argschema.fields.InputFile,
        cli_as_single_argument=True,
        required=True,
        description='Set of paths to load model_inputs from. Each json file '
                    'has schema given by '
                    '`deepcell.cli.schemas.data.ModelInputSchema`. Multiple '
                    'files can be specified in the case of "CV"'
    )
    experiment_id = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description='If provided, we are running inference on a specific '
                    'experiment'
    )
    mode = argschema.fields.String(
        required=True,
        validate=OneOf(('CV', 'test', 'production')),
        description='Inference mode. "CV" gathers predictions on a validation '
                    'set. "test" gathers predictions on a test set.'
                    '"production" gathers predictions on a set of examples in '
                    'production where there are no labels'
    )
    model_params = argschema.fields.Nested(
        ModelSchema,
        default={}
    )
