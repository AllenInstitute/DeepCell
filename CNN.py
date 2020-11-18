from torch import nn


class CNN(nn.Module):
    def __init__(self, cfg, batch_norm=True, batch_norm_before_nonlin=True, num_classes=1,
                 dropout_prob=0.5):
        super().__init__()
        self.features = self._make_layers(cfg=cfg, batch_norm=batch_norm,
                                          batch_norm_before_nonlin=batch_norm_before_nonlin)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        last_conv_filter_num = cfg[-2]
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_filter_num * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_layers(cfg, batch_norm=True, batch_norm_before_nonlin=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    if batch_norm_before_nonlin:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

