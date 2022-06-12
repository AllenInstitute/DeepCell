import json

import argschema
import marshmallow

from deepcell.cli.schemas.data import ModelInputSchema, DataSchema


class BaseSchema(argschema.ArgSchema):
    batch_size = argschema.fields.Int(
        default=64,
        description='Batch size'
    )
    save_path = argschema.fields.OutputDir(
        default='/tmp/model',
        description='Where to save model and all outputs. On AWS, this is '
                    'ignored and the model artifacts are rather saved on S3.'
    )
    data_params = argschema.fields.Nested(
        DataSchema,
        default={}
    )
    model_load_path = argschema.fields.InputDir(
        default=None,
        allow_none=True,
        description='Path to load a model checkpoint. If training, this will '
                    'continue training using this checkpoint. If inference, '
                    'will use this checkpoint for inference'
    )
    log_path = argschema.fields.OutputFile(
        default=None,
        allow_none=True,
        description='path to write log'
    )

    @marshmallow.post_load
    def model_inputs(self, data):
        for key in ('model_inputs_path', 'train_model_inputs_path',
                    'validation_model_inputs_path'):
            if key not in data:
                continue
            elif data[key] is None:
                data[key.replace('_path', '')] = None
            else:
                with open(str(data[key]), 'r') as f:
                    model_inputs = json.load(f)
                model_inputs = [ModelInputSchema().load(model_input)
                                for model_input in model_inputs]
                data[key.replace('_path', '')] = model_inputs
        return data
