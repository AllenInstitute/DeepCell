from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from deepcell.models.util import get_last_filter_num


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, localization_feature_extractor: List):
        super().__init__()
        self.localization_feature_extractor = torch.nn.Sequential(
            *localization_feature_extractor,
            torch.nn.AdaptiveAvgPool2d((7, 7))
        )

        for layer in self.localization_feature_extractor:
            for p in layer.parameters():
                p.requires_grad = False

        last_conv_filter_num = get_last_filter_num(
            layers=self.localization_feature_extractor)
        in_features = last_conv_filter_num * 7 * 7

        self.localization_regressor = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=32),
            nn.ReLU(inplace=True),

            # out_features = 3 because using this transformation matrix
            # [[s 0 tx]
            #  [0 s ty]]
            # which has 3 parameters
            # allows cropping, translation, and isotropic scaling
            nn.Linear(in_features=32, out_features=3)
        )

        # Initialize the weights/bias with identity transformation
        self.localization_regressor[-1].weight.data.zero_()
        self.localization_regressor[-1].bias.data.copy_(
            torch.tensor([1, 0, 0], dtype=torch.float))

    def forward(self, x):
        input = x

        x = self.localization_feature_extractor(x)
        x = x.reshape(x.size(0), -1)
        x = self.localization_regressor(x)

        s = x[:, 0]
        tx = x[:, 1]
        ty = x[:, 2]

        A = torch.tensor([[
            [s, 0, tx],
            [0, s, ty]
        ] for s, tx, ty in zip(s, tx, ty)], dtype=torch.float)

        grid = F.affine_grid(A, input.size())
        x = F.grid_sample(input, grid)
        return x
