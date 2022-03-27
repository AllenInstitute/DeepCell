import json

import argschema
import marshmallow

from deepcell.cli.schemas.data import DataSchema, ModelInputSchema
from deepcell.cli.schemas.model import TrainModelSchema
from deepcell.cli.schemas.optimization import OptimizationSchema
from deepcell.cli.schemas.tracking import TrackingSchema


class _TrainSchema(argschema.ArgSchema):
    model_params = argschema.fields.Nested(
        TrainModelSchema,
        default={}
    )
    optimization_params = argschema.fields.Nested(
        OptimizationSchema,
        default={}
    )
    data_params = argschema.fields.Nested(
        DataSchema,
        default={}
    )
    tracking_params = argschema.fields.Nested(
        TrackingSchema,
        default={}
    )
    save_path = argschema.fields.OutputDir(
        default='/tmp/model',
        description='Where to save model and all outputs. Only used in '
                    'local mode. Otherwise the model and artifacts are saved '
                    'to the default sagemaker path on s3'
    )
    model_load_path = argschema.fields.InputDir(
        default=None,
        allow_none=True,
        description='Path to load a model checkpoint to continue training'
    )
    batch_size = argschema.fields.Int(
        default=64,
        description='Batch size'
    )

    @marshmallow.post_load
    def model_inputs(self, data):
        for key in ('model_inputs_path', 'train_model_inputs_path',
                    'validation_model_inputs_path'):
            if key not in data:
                continue

            with open(str(data[key]), 'r') as f:
                model_inputs = json.load(f)
            model_inputs = [ModelInputSchema().load(model_input)
                            for model_input in model_inputs]
            data[key.replace('_path', '')] = model_inputs
        return data


class TrainSchema(_TrainSchema):
    train_model_inputs_path = argschema.fields.InputFile(
        required=True,
        description='Path to json file for training examples where each '
                    'instance has schema given by '
                    '`deepcell.cli.schemas.data.ModelInputSchema`'
    )
    validation_model_inputs_path = argschema.fields.InputFile(
        required=True,
        description='Path to json file for validation examples where each '
                    'instance has schema given by '
                    '`deepcell.cli.schemas.data.ModelInputSchema`'
    )
    fold = argschema.fields.Int(
        default=None,
        allow_none=True,
        description='Which fold is being trained. Leave as None if not '
                    'training using kfold CV'
    )


class KFoldCrossValidationSchema(_TrainSchema):
    model_inputs_path = argschema.fields.InputFile(
        required=True,
        description='Path to json file for training examples where each '
                    'instance has schema given by '
                    '`deepcell.cli.schemas.data.ModelInputSchema`'
    )
    n_folds = argschema.fields.Int(
        default=5,
        description='Number of folds for cross validation'
    )
