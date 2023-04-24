from typing import List, Tuple, Optional

import pandas as pd
import torch
import numpy as np

# This private module provides useful transforms for video.
# Available as of torchvision 0.15
# Easier than reimplementing here
import torchvision.transforms._transforms_video as transforms_video

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.video import S3D_Weights

from deepcell.util.construct_dataset.vote_tallying_strategy import \
    VoteTallyingStrategy
from deepcell.util.construct_dataset.construct_dataset_utils import construct_dataset
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.transforms import RandomRotate90
from deepcell.transform import Transform


class RoiDataset(Dataset):
    def __init__(self,
                 model_inputs: List[ModelInput],
                 image_dim=(128, 128),
                 transform: Transform = None,
                 debug=False,
                 mask_out_projections=False,
                 center_roi_centroid=False,
                 centroid_brightness_quantile=0.8,
                 centroid_use_mask=False,
                 weight_samples_by_labeler_confidence: bool = False,
                 cell_labeling_app_host: Optional[str] = None
                 ):
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
            weight_samples_by_labeler_confidence
                Weight each sample by labeler confidence in the
                training/validation loss function
            cell_labeling_app_host
                Needed to pull labels if weight_samples_by_labeler_confidence
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

        if weight_samples_by_labeler_confidence:
            if cell_labeling_app_host is None:
                raise ValueError('need cell_labeling_app_host if '
                                 'weight_samples_by_labeler_confidence')
            self._sample_weights = self._get_labeler_agreement(
                cell_labeling_app_host=cell_labeling_app_host)
        else:
            self._sample_weights = None

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

        input = np.load(str(obs.path))

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

        if self._sample_weights is not None:
            sample_weight = self._sample_weights.loc[
                (obs.experiment_id, obs.roi_id)]['agreement']
        else:
            sample_weight = None

        return input, target, sample_weight

    def __len__(self):
        return len(self._model_inputs)

    @staticmethod
    def _get_labeler_agreement(
        cell_labeling_app_host: str
    ) -> pd.DataFrame:
        """For each ROI, calculates labeler agreement with the majority label
        """
        raw_labels = construct_dataset(
            cell_labeling_app_host=cell_labeling_app_host,
            raw=True
        )
        majority_labels = construct_dataset(
            cell_labeling_app_host=cell_labeling_app_host,
            vote_tallying_strategy=VoteTallyingStrategy.MAJORITY
        )
        majority_labels = majority_labels.set_index(
            ['experiment_id', 'roi_id'])
        agreement = raw_labels.groupby(['experiment_id', 'roi_id'])['label'] \
            .apply(lambda x: (
                x == majority_labels.loc[x.name]['label']).mean()) \
            .reset_index() \
            .rename(columns={'label': 'agreement'})

        agreement = agreement.set_index(['experiment_id', 'roi_id'])
        return agreement
