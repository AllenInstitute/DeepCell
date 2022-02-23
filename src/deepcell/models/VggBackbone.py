import numpy as np
import torch
import torchvision
from torch import nn


class VggBackbone(torch.nn.Module):
    def __init__(self, model, truncate_to_layer, classifier_cfg,
                 dropout_prob=0.5, freeze_up_to_layer=None,
                 shuffle_final_activation_map=False,
                 final_activation_map_spatial_dimensions=(1, 1)):
        super().__init__()
        conv_layers = self._truncate_to_layer(model=model, layer=truncate_to_layer)
        self.features = torch.nn.Sequential(*conv_layers)

        for layer in self.features[:freeze_up_to_layer]:
            for p in layer.parameters():
                p.requires_grad = False

        self.avgpool = torch.nn.AdaptiveAvgPool2d(
            final_activation_map_spatial_dimensions)

        last_conv_filter_num = self._get_last_filter_num()
        in_features = last_conv_filter_num * np.prod(
            final_activation_map_spatial_dimensions)
        self.classifier = self._make_classifier_layers(cfg=classifier_cfg, in_features=in_features,
                                                       dropout_prob=dropout_prob)
        self._shuffle_final_activation_map = shuffle_final_activation_map

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        if self._shuffle_final_activation_map:
            # Shuffle spatial dimensions of activation map
            x = x.reshape(x.size(0), x.size(1), x.shape[-2] * x.shape[-1])
            indexes = torch.randperm(x.shape[2])
            x = x[:, :, indexes]

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


if __name__ == '__main__':
    cnn = torchvision.models.vgg11_bn(pretrained=True, progress=False)
    cnn = VggBackbone(model=cnn, truncate_to_layer=22, classifier_cfg=[512],
                      freeze_up_to_layer=15)
    pass