import copy

import torch
from torch import nn

from deepcell.models.util import truncate_model_to_layer, get_last_filter_num
from deepcell.models.spatial_transformer_network import \
    SpatialTransformerNetwork


class VggBackbone(torch.nn.Module):
    def __init__(self, model, truncate_to_layer, classifier_cfg,
                 dropout_prob=0.5, freeze_up_to_layer=None,
                 use_spatial_transformer_network=False):
        super().__init__()
        conv_layers = truncate_model_to_layer(
            model=model, layer=truncate_to_layer)
        self.features = torch.nn.Sequential(*conv_layers)

        for layer in self.features[:freeze_up_to_layer]:
            for p in layer.parameters():
                p.requires_grad = False

        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        last_conv_filter_num = get_last_filter_num(layers=self.features)
        in_features = last_conv_filter_num * 7 * 7
        self.classifier = self._make_classifier_layers(cfg=classifier_cfg, in_features=in_features,
                                                       dropout_prob=dropout_prob)

        if use_spatial_transformer_network:
            self._stn = SpatialTransformerNetwork(
                localization_feature_extractor=copy.deepcopy(conv_layers))
        else:
            self._stn = None

    def forward(self, x):
        if self._stn is not None:
            x = self._stn(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_classifier_layers(cfg, in_features, dropout_prob, num_classes=1):
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