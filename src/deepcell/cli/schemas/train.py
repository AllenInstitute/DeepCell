import argschema

from deepcell.cli.schemas.data import DataSchema
from deepcell.cli.schemas.model import TrainModelSchema
from deepcell.cli.schemas.optimization import OptimizationSchema


class TrainSchema(argschema.ArgSchema):
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
        required=True
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
    test_fraction = argschema.fields.Float(
        default=0.3,
        description='Fraction of data to reserve for the test set'
    )
    batch_size = argschema.fields.Int(
        default=64,
        description='Batch size'
    )
    n_folds = argschema.fields.Int(
        default=5,
        description='Number of folds for cross validation'
    )
