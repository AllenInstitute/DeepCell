import argschema
from marshmallow.validate import OneOf


class SchedulerSchema(argschema.ArgSchema):
    type = argschema.fields.String(
        validate=OneOf((None, 'ReduceLROnPlateau')),
        allow_none=True,
        default='ReduceLROnPlateau',
        description='Learning rate decay algorithm. If ReduceLROnPlateau, '
                    'then monitor the validation loss, and reduce the '
                    'learning rate when it has plateaued. If None '
                    'then it is disabled'
    )
    patience = argschema.fields.Int(
        default=15,
        description='Scheduler patience (number of iterations before the '
                    'scheduler reduces the learning rate)'
    )
    factor = argschema.fields.Float(
        default=0.5,
        description='Factor by which to reduce learning rate'
    )
