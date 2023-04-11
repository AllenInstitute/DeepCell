from typing import List, Tuple

import torch
from PIL import Image
import numpy as np

# This private module provides useful transforms for video.
# Available as of torchvision 0.15
# Easier than reimplementing here
import torchvision.transforms._transforms_video as transforms_video

from deepcell.datasets.channel import Channel
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.video import S3D_Weights

from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.transforms import RandomRotate90
from deepcell.datasets.util import center_roi
from deepcell.transform import Transform
from deepcell.util import get_experiment_genotype_map


class RoiDataset(Dataset):
    def __init__(self,
                 model_inputs: List[ModelInput],
                 image_dim=(128, 128),
                 transform: Transform = None,
                 debug=False,
                 cre_line=None,
                 mask_out_projections=False,
                 center_roi_centroid=False,
                 centroid_brightness_quantile=0.8,
                 centroid_use_mask=False):
        """
        A dataset of segmentation masks as identified by Suite2p with
        binary label "cell" or "not cell"
        Args:
            model_inputs:
                A set of records in this dataset
            image_dim:
                Dimensions of artifacts
            transform:
                Transform to apply
            debug:
                Whether in debug mode. This will limit the number of
                records in the dataset to only a single example from each class
            cre_line:
                Whether to limit to a cre_line
            mask_out_projections:
                Whether to mask out projections to only include pixels in
                the mask
            center_roi_centroid
                The classifier has poor performance with a soma that is not
                centered in frame. Find the ROI centroid and use that to
                center in the frame.
            centroid_brightness_quantile
                Used if center_roi_centroid is True. The quantile to use when
                zeroing out dim pixels. Used to
                focus centroid on soma, which is brighter.
            centroid_use_mask
                Used if center_roi_centroid is True. See `use_mask` arg in
                `deepcell.datasets.util.calc_roi_centroid`
        """
        super().__init__()

        self._model_inputs = model_inputs
        self._image_dim = image_dim
        self.transform = transform
        self._mask_out_projections = mask_out_projections
        self._y = self.get_numeric_labels(model_inputs=model_inputs)
        self._center_roi_centroid = center_roi_centroid
        self._centroid_brightness_quantile = centroid_brightness_quantile
        self._centroid_use_mask = centroid_use_mask

        if cre_line:
            experiment_genotype_map = get_experiment_genotype_map()
            self._filter_by_cre_line(
                experiment_genotype_map=experiment_genotype_map,
                cre_line=cre_line)

        if debug:
            not_cell_idx = np.argwhere(self._y == 0)[0][0]
            cell_idx = np.argwhere(self._y == 1)[0][0]
            self._model_inputs = [
                self._model_inputs[not_cell_idx], self._model_inputs[cell_idx]]
            self._y = np.array([0, 1])

    @property
    def model_inputs(self) -> List[ModelInput]:
        return self._model_inputs

    @property
    def y(self) -> np.ndarray:
        return self._y

    @staticmethod
    def get_default_transforms(
            crop_size: Tuple[int, int],
            is_train: bool,
            means: List[float] = None,
            stds: List[float] = None
    ) -> Transform:
        """
        Gets the default transforms

        Parameters
        ----------
        crop_size
            Crop size
        is_train
            Whether this is a train or test dataset
        means
            Channel-wise means to standardize data
            (after converting to [0, 1] range).
            The defaults are the channel-wise means on kinetics-400
        stds
            Channel-wise stds to standardize data
            (after converting to [0, 1] range)
            The defaults are the channel-wise stds on kinetics-400
        Returns
        -------
        Transform
        """
        # loading the data distribution from a pretrained video model
        # in case means and stds not passed
        if means is None:
            means = S3D_Weights.DEFAULT.transforms().mean
        if stds is None:
            stds = S3D_Weights.DEFAULT.transforms().std

        if is_train:
            all_transform = transforms.Compose([
                lambda x: torch.tensor(x),
                transforms_video.ToTensorVideo(),
                RandomRotate90(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.CenterCrop(size=crop_size),
                transforms_video.NormalizeVideo(mean=means, std=stds)
            ])

            return Transform(all_transform=all_transform)
        else:
            all_transform = transforms.Compose([
                lambda x: torch.tensor(x),
                transforms_video.ToTensorVideo(),
                transforms.CenterCrop(size=crop_size),
                transforms_video.NormalizeVideo(mean=means, std=stds)
            ])

            return Transform(all_transform=all_transform)

    @staticmethod
    def get_numeric_labels(model_inputs: List[ModelInput]) -> np.ndarray:
        """
        Returns np.ndarray of labels encoded as int

        Parameters
        ----------
        model_inputs: list of ModelInput

        Returns
        -------
        np array of labels encoded as int
        """
        return np.array([int(x.label == 'cell') for x in model_inputs])

    def __getitem__(self, index):
        obs = self._model_inputs[index]

        input = self._construct_input(obs=obs)

        if self.transform:
            if (
                    self.transform.avg_transform or
                    self.transform.max_transform or
                    self.transform.mask_transform):
                raise NotImplementedError('Transform on individual channels '
                                          'not supported anymore')

            if self.transform.all_transform:
                input = self.transform.all_transform(input)

        # TODO collate_fn should be used instead
        target = self._y[index]

        return input, target

    def __len__(self):
        return len(self._model_inputs)

    def _construct_input(self, obs: ModelInput) -> np.ndarray:
        """
        Construct a single input

        Returns:
            A numpy array of type uint8 and shape *dim, n_channels
        """
        n_channels = len(obs.channel_order)
        res = np.zeros((*self._image_dim, n_channels), dtype=np.uint8)

        for i, channel in enumerate(obs.channel_order):
            path = obs.channel_path_map[channel]

            with open(path, 'rb') as f:
                img = Image.open(f)
                img = np.array(img)
            if channel == Channel.MASK:
                try:
                    res[:, :, i] = img
                except ValueError:
                    # TODO fix this issue
                    pass
            else:
                res[:, :, i] = img

        if self._mask_out_projections:
            # making plural since in theory could be multiple channels
            masks = [i for i, channel in enumerate(obs.channel_order)
                     if channel == Channel.MASK]
            if len(masks) != 0:
                for i, channel_name in enumerate(obs.channel_order):
                    if channel_name != Channel.MASK:
                        res[:, :, i][np.where(masks[0] == 0)] = 0

        if self._center_roi_centroid:
            res = center_roi(
                image=res, brightness_quantile=self._centroid_brightness_quantile,
                use_mask=self._centroid_use_mask
            )

        return res

    def _filter_by_cre_line(self, experiment_genotype_map, cre_line):
        """
        Modifies dataset by filtering by creline

        Args:
            experiment_genotype_map:
                Maps experiment to genotype
            cre_line:
                Cre_line to filter by
        Returns:
            None, modifies inplace
        """
        self._model_inputs = [
            x for x in self._model_inputs if experiment_genotype_map[
                x.experiment_id].startswith(cre_line)]
