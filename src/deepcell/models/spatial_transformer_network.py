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
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # out_features = 3 because using this transformation matrix
            # [[s 0 tx]
            #  [0 s ty]]
            # which has 3 parameters
            # allows cropping, translation, and isotropic scaling
            nn.Linear(in_features=512, out_features=3)
        )

        # Initialize the weights/bias with identity transformation
        self.localization_regressor[-1].weight.data.zero_()
        self.localization_regressor[-1].bias.data.copy_(
            torch.tensor([0.5, 0, 0], dtype=torch.float))

    def forward(self, x):
        input = x

        x = self.localization_feature_extractor(x)
        x = x.reshape(x.size(0), -1)
        theta = self.localization_regressor(x)

        scale = theta[:, 0].unsqueeze(1)
        scale_mat = torch.cat((scale, scale), 1)
        translation = theta[:, 1:].unsqueeze(-1)
        A = torch.cat((torch.diag_embed(scale_mat),
                       translation), -1)

        # s = theta[:, 0]
        # tx = theta[:, 1]
        # ty = theta[:, 2]
        #
        # A = torch.tensor([[
        #     [s, 0, tx],
        #     [0, s, ty]
        # ] for s, tx, ty in zip(s, tx, ty)], dtype=torch.float,
        #     requires_grad=True)

        grid = F.affine_grid(A, [x.size(0), x.size(1), 60, 60],
                             align_corners=False)
        if torch.cuda.is_available():
            grid = grid.cuda()
        x = F.grid_sample(input, grid, align_corners=False)
        return x
