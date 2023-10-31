import os.path
from pathlib import Path

import argschema
import marshmallow
from marshmallow.validate import OneOf

from deepcell.datasets.channel import Channel
from deepcell.datasets.model_input import ModelInput


class ChannelField(argschema.fields.String):
    def _validate(self, value):
        supported_channels = [x.value for x in Channel]
        if value not in supported_channels:
            raise ValueError(f'Only {supported_channels} supported. '
                             f'You passed {value}')


class ModelInputSchema(argschema.ArgSchema):
    """Defines settings for a single model input"""
    experiment_id = argschema.fields.String(
        required=True,
        description='experiment_id'
    )
    roi_id = argschema.fields.Int(
        required=True,
        description='roi id'
    )
    channel_order = argschema.fields.List(
        ChannelField(),
        required=True,
        description='Channels to use as input'
    )
    channel_path_map = argschema.fields.Dict(
        keys=ChannelField(),
        values=argschema.fields.InputFile(),
        description='Map between channel and path'
    )
    label = argschema.fields.String(
        default=None,
        allow_none=True,
        description='label. Null in case of inference.',
        validate=OneOf((None, 'cell', 'not cell'))
    )
    project_name = argschema.fields.String(
        required=True,
        description='Project name'
    )

    @marshmallow.post_load
    def make_model_input(self, data, **kwargs):
        data = {k: v for k, v in data.items() if k not in ('log_level',)}
        for k, v in data.items():
            if isinstance(v, str):
                if os.path.isfile(v):
                    data[k] = Path(v)

        # serialize to Channel
        data['channel_order'] = [
            getattr(Channel, x) for x in data['channel_order']]
        channel_path_map = {
            getattr(Channel, c): p
            for c, p in data['channel_path_map'].items()}
        data['channel_path_map'] = channel_path_map

        return ModelInput(**data)


class DataSchema(argschema.ArgSchema):
    """Defines settings for model inputs"""
    crop_size = argschema.fields.Tuple(
        (argschema.fields.Int, argschema.fields.Int),
        default=(128, 128),
        description='Width, height center crop size to apply to all inputs'
    )

    center_roi_centroid = argschema.fields.Bool(
        default=False,
        description='Manually center input by finding centroid of soma'
    )
    channel_wise_means = argschema.fields.List(
        argschema.fields.Float,
        default=[0.485, 0.456, 0.406],
        cli_as_single_argument=True,
        description='Channel wise means to standardize data '
                    '(after converting to [0, 1] range). '
                    'The defaults are the ImageNet published channel-wise '
                    'means. Only use these means if doing transfer learning '
                    'on a model trained on ImageNet. Otherwise, use dataset '
                    'means'
    )
    channel_wise_stds = argschema.fields.List(
        argschema.fields.Float,
        default=[0.229, 0.224, 0.225],
        cli_as_single_argument=True,
        description='Channel wise stds to standardize data '
                    '(after converting to [0, 1] range). '
                    'The defaults are the ImageNet published channel-wise '
                    'stds. Only use these stds if doing transfer learning '
                    'on a model trained on ImageNet. Otherwise, use dataset '
                    'stds'
    )
