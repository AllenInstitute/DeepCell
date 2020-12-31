from torch import nn


class RNNDecoder(nn.Module):
    def __init__(self, classifier_cfg, input_dim=300, num_layers=3, hidden_size=256, dropout_prob=0.5):
        super().__init__()

        self.dropout_prob = dropout_prob

        self.LSTM = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.hidden_state = None

        self.classifier = self._make_classifier_layers(cfg=classifier_cfg, in_features=hidden_size)

    def forward(self, x):
        x, self.hidden_state = self.LSTM(x, self.hidden_state)

        # extract output at last time step (of size hidden_size)
        x = x[:, -1]

        x = self.classifier(x)

        return x

    def detach_hidden_state(self):
        self.hidden_state = tuple(v.detach() for v in self.hidden_state)

    @staticmethod
    def _make_classifier_layers(cfg, in_features, dropout_prob=0.0, num_classes=1):
        cfg.insert(0, in_features)
        layers = []
        for i, v in enumerate(cfg[1:], start=1):
            layers += [
                nn.Linear(cfg[i - 1], cfg[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            ]

        layers.append(nn.Linear(cfg[-1], num_classes))
        return nn.Sequential(*layers)
