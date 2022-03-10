import argschema

from deepcell.cli.schemas.early_stopping import EarlyStoppingSchema
from deepcell.cli.schemas.scheduler import SchedulerSchema


class OptimizationSchema(argschema.ArgSchema):
    use_learning_rate_decay = argschema.fields.Bool(
        default=True,
        description='Whether to use learning rate decay. '
                    'Currently only ReduceLROnPlateau supported'
    )
    n_epochs = argschema.fields.Int(
        default=1000,
        description='Number of epochs to train for'
    )
    learning_rate = argschema.fields.Float(
        default=1e-4,
        description='Initial Learning rate'
    )
    weight_decay = argschema.fields.Float(
        default=0.0,
        description='L2 regularization strength'
    )
    scheduler_params = argschema.fields.Nested(
        SchedulerSchema,
        default={}
    )
    early_stopping_params = argschema.fields.Nested(
        EarlyStoppingSchema,
        default={}
    )
