from typing import List, Optional

import numpy as np
import torch
from torch import nn


class Classifier(torch.nn.Module):
    """A classifier using a CNN backbone"""

    def __init__(self, model: torch.nn.Module, truncate_to_layer: int,
                 classifier_cfg: List[int],
                 dropout_prob: float = 0.5,
                 freeze_up_to_layer: Optional[int] = None,
                 final_activation_map_spatial_dimensions=(1, 1)):
        """

        Args:
            model:
                The pytorch model. The classifier will be replaced with a new
                one and the CNN layers may be truncated using
                `truncate_to_layer`
            truncate_to_layer:
                Index of the layer of `model` to truncate to. It is possible
                to use too big of a model. `truncate_to_layer` helps to shrink
                the model capacity.
            classifier_cfg:
                A configuration of the form
                [# neurons in first layer, ...# neurons in last hidden layer]
                An empty list means to use a linear classifier
            dropout_prob:
                Dropout probability for fully connected layers
            freeze_up_to_layer:
                Use for finetuning certain layers in the network. When using
                a pretrained model, the first few layers encode fundamental
                properties of images such as edges. The later layers encode
                domain information. So one might want to keep the first few
                layers but finetune the later layers. This argument controls
                that
            final_activation_map_spatial_dimensions:
                The final activation map spatial dimension before flattening
                into a vector for input into the classifier.
        """
        super().__init__()
        conv_layers = self._truncate_to_layer(model=model,
                                              layer=truncate_to_layer)
        self.features = torch.nn.Sequential(*conv_layers)

        for layer in self.features[:freeze_up_to_layer]:
            for p in layer.parameters():
                p.requires_grad = False

        self.avgpool = torch.nn.AdaptiveAvgPool2d(
            final_activation_map_spatial_dimensions)

        last_conv_filter_num = self._get_last_filter_num()
        in_features = last_conv_filter_num * np.prod(
            final_activation_map_spatial_dimensions)
        self.classifier = self._make_classifier_layers(
            cfg=classifier_cfg,
            in_features=in_features,
            dropout_prob=dropout_prob)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _truncate_to_layer(model, layer):
        conv_layers = list(model.children())[0]
        return conv_layers[:layer]

    def _get_last_filter_num(self):
        idx = -1
        while idx > -1 * len(self.features):
            if hasattr(self.features[idx], 'out_channels'):
                return self.features[idx].out_channels
            idx -= 1

        raise Exception('Could not find number of filters in last conv layer')

    @staticmethod
    def _make_classifier_layers(cfg, in_features, dropout_prob, num_classes=1):
        if len(cfg) == 0:
            return nn.Sequential(
                nn.Linear(in_features=in_features, out_features=num_classes)
            )

        layers = [
            nn.Linear(in_features=in_features, out_features=cfg[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        ]
        for i, v in enumerate(cfg[1:], start=1):
            layers += [
                nn.Linear(cfg[i - 1], cfg[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            ]

        layers.append(nn.Linear(cfg[-1], num_classes))
        return nn.Sequential(*layers)
