import argschema

from deepcell.cli.schemas.base import BaseSchema
from deepcell.cli.schemas.model import TrainModelSchema
from deepcell.cli.schemas.optimization import OptimizationSchema
from deepcell.cli.schemas.tracking import TrackingSchema


class _TrainSchema(BaseSchema):
    model_params = argschema.fields.Nested(
        TrainModelSchema,
        default={}
    )
    optimization_params = argschema.fields.Nested(
        OptimizationSchema,
        default={}
    )
    tracking_params = argschema.fields.Nested(
        TrackingSchema,
        default={}
    )


class TrainSchema(_TrainSchema, BaseSchema):
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


class KFoldCrossValidationSchema(_TrainSchema, BaseSchema):
    n_folds = argschema.fields.Int(
        default=5,
        description='Number of folds for cross validation'
    )
