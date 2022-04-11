import argschema
from marshmallow.validate import OneOf

from argschema.schemas import DefaultSchema


class ModelSchema(DefaultSchema):
    model_architecture = argschema.fields.String(
        validation=OneOf(('vgg11_bn', )),
        default='vgg11_bn',
        description='One of the well known model architectures in the '
                    'torchvision model zoo (ie one of these '
                    'https://pytorch.org/vision/stable/models.html)'
    )
    use_pretrained_model = argschema.fields.Bool(
        default=True,
        description='Whether to use a model pretrained on imagenet'
    )
    truncate_to_layer = argschema.fields.Int(
        default=None,
        allow_none=True,
        description='Layer index to truncate model to'
    )
    classifier_cfg = argschema.fields.List(
        argschema.fields.Int,
        cli_as_single_argument=True,
        default=[],
        description='A configuration of the form '
                    '[# neurons in first layer, '
                    '...# neurons in last hidden layer]. An empty list means '
                    'to use a linear classifier'
    )
    final_activation_map_spatial_dimensions = argschema.fields.Tuple(
        (argschema.fields.Int, argschema.fields.Int),
        default=(1, 1),
        description='The final activation map spatial dimension before flattening '
                    'into a vector for input into the classifier'
    )


class TrainModelSchema(ModelSchema):
    freeze_to_layer = argschema.fields.Int(
        default=None,
        allow_none=True,
        description='Layer index up to which will be frozen (not optimized). '
                    'Use for finetuning certain layers in the network. When '
                    'using a pretrained model, the first few layers encode '
                    'fundamental properties of images such as edges. The '
                    'later layers encode domain information. So one might '
                    'want to keep the first few layers but finetune the later '
                    'layers. This argument controls that'
    )
    dropout_prob = argschema.fields.Float(
        default=0.0,
        description='Dropout probability for fully connected layers'
    )
