import torch
from torch import nn


class CNNEncoder(torch.nn.Module):
    def __init__(self, model, truncate_to_layer, embedding_cfg, dropout_prob=0.5, freeze_layers=True, embed_dim=300):
        super().__init__()
        conv_layers = self._truncate_to_layer(model=model, layer=truncate_to_layer)
        features = torch.nn.Sequential(*conv_layers)

        if freeze_layers:
            for p in self.features.parameters():
                p.requires_grad = False

        avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        self.features = torch.nn.Sequential(features, avgpool)

        last_conv_filter_num = self._get_last_filter_num()
        in_features = last_conv_filter_num * 7 * 7
        self.embedding = self._make_embedding_layers(cfg=embedding_cfg, in_features=in_features,
                                                     dropout_prob=dropout_prob, embed_dim=embed_dim)

    def forward(self, x):
        cnn_embed_seq = []

        num_frames = x.size(2)
        print(f'tensor shape: {x.shape}')

        for f in range(num_frames):
            x = self.features(x[:, :, f])
            x = x.reshape(x.size(0), -1)    # flatten
            x = self.embedding(x)

            cnn_embed_seq.append(x)

        # (time dim, sample dim, latent dim) -> (sample dim, time dim, latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq

    @staticmethod
    def _truncate_to_layer(model, layer):
        conv_layers = list(model.children())[0]
        return conv_layers[:layer]

    def _get_last_filter_num(self):
        features = self.features[0]
        idx = -1
        while idx > -1 * len(features):
            if hasattr(features[idx], 'out_channels'):
                return features[idx].out_channels
            idx -= 1

        raise Exception('Could not find number of filters in last conv layer')

    @staticmethod
    def _make_embedding_layers(cfg, in_features, embed_dim, dropout_prob=0.0):
        cfg.insert(0, in_features)
        layers = []
        for i, v in enumerate(cfg[1:], start=1):
            layers += [
                nn.Linear(cfg[i - 1], cfg[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            ]

        layers.append(nn.Linear(cfg[-1], embed_dim))
        return nn.Sequential(*layers)