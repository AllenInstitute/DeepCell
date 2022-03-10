import argschema
from marshmallow.validate import OneOf


class EarlyStoppingSchema(argschema.ArgSchema):
    patience = argschema.fields.Int(
        default=30,
        description='Number of epochs to activate early stopping'
    )
    monitor = argschema.fields.String(
        default='f1',
        validate=OneOf(('loss', 'f1')),
        description='Metric to monitor in order to activate early stopping'
    )
