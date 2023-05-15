import random
from typing import List, Tuple, Optional

import cv2
import h5py
import pandas as pd
import torch
import numpy as np

# This private module provides useful transforms for video.
# Available as of torchvision 0.15
# Easier than reimplementing here
import torchvision.transforms._transforms_video as transforms_video
from ophys_etl.types import OphysROI
from ophys_etl.utils.array_utils import normalize_array, get_cutout_indices, \
    get_cutout_padding

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.video import S3D_Weights

from deepcell.util.construct_dataset.vote_tallying_strategy import \
    VoteTallyingStrategy
from deepcell.util.construct_dataset.construct_dataset_utils import construct_dataset
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.transforms import RandomRotate90, ReverseVideo
from deepcell.transform import Transform


class _EmptyMaskException(Exception):
    """Raised if mask is empty"""
    pass


class RoiDataset(Dataset):
    def __init__(self,
                 model_inputs: List[ModelInput],
                 is_train: bool,
                 image_dim=(128, 128),
                 transform: Transform = None,
                 debug=False,
                 mask_out_projections=False,
                 center_roi_centroid=False,
                 centroid_brightness_quantile=0.8,
                 centroid_use_mask=False,
                 weight_samples_by_labeler_confidence: bool = False,
                 cell_labeling_app_host: Optional[str] = None,
                 n_frames: int = 16,
                 temporal_downsampling_factor: int = 1,
                 test_use_highest_peak: bool = False,
                 limit_to_n_highest_peaks: Optional[int] = 5,
                 fov_shape: Tuple[int, int] = (512, 512)
                 ):
        """
        A dataset of segmentation masks as identified by Suite2p with
        binary label "cell" or "not cell"
        Args:
            model_inputs:
                A set of records in this dataset
            is_train
                Whether we are training or doing inference. If training,
                we will randomly sample a peak to construct the frames
                If inference, we will use a specific peak
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
            n_frames
                Number of frames to use for the clip
            temporal_downsampling_factor
                Evenly sample every nth frame so that we
                have `n_frames` total frames. I.e. If it is 4, then
                we start with 4 * `n_frames` frames around peak and
                sample every 4th frame
            test_use_highest_peak
                If this is a test set, whether to use the highest peak from
                the list of peaks for this ROI. If not, we will get predictions
                for all peaks and average the results
            limit_to_n_highest_peaks
                For testing, only sample the n highest peaks
            fov_shape
                FOV shape
        """
        super().__init__()

        if not is_train:
            if test_use_highest_peak:
                for model_input in model_inputs:
                    model_input.peak = model_input.get_n_highest_peaks(n=1)[0]

        self._is_train = is_train
        self._model_inputs = model_inputs
        self._image_dim = image_dim
        self.transform = transform
        self._mask_out_projections = mask_out_projections
        self._y = self.get_numeric_labels(model_inputs=model_inputs)
        self._center_roi_centroid = center_roi_centroid
        self._centroid_brightness_quantile = centroid_brightness_quantile
        self._centroid_use_mask = centroid_use_mask
        self._n_frames = n_frames
        self._temporal_downsampling_factor = temporal_downsampling_factor
        self._limit_to_n_highest_peaks = limit_to_n_highest_peaks
        self._fov_shape = fov_shape

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
                ReverseVideo(p=0.5),
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

        if obs.peaks is None:
            raise ValueError('Expected the model_input to contain peaks')
        if self._limit_to_n_highest_peaks is not None:
            peaks = obs.get_n_highest_peaks(
                n=self._limit_to_n_highest_peaks)
        else:
            peaks = obs.peaks

        n_frames = self._n_frames * self._temporal_downsampling_factor
        nframes_before_after = int(n_frames/2)

        row_indices = get_cutout_indices(
            center_dim=obs.roi.bounding_box_center_y,
            image_dim=self._fov_shape[0],
            cutout_dim=self._image_dim[0])
        col_indices = get_cutout_indices(
            center_dim=obs.roi.bounding_box_center_x,
            image_dim=self._fov_shape[0],
            cutout_dim=self._image_dim[0])

        with h5py.File(obs.ophys_movie_path, 'r') as f:
            mov_len = f['data'].shape[0]

        peaks = sorted(peaks, key=lambda x: x.peak)
        frame_idxs = []
        for peak in peaks:
            start_index = max(0, peak.peak - nframes_before_after)
            end_index = min(mov_len, peak.peak + nframes_before_after)
            if self._n_frames == 1:
                end_index += 1
            idxs = np.arange(start_index, end_index,
                             self._temporal_downsampling_factor)
            frame_idxs += idxs.tolist()

        # h5py doesn't allow an index to be repeated
        frame_idxs = list(set(frame_idxs))

        frame_idxs = sorted(frame_idxs)

        with h5py.File(obs.ophys_movie_path, 'r') as f:
            frames = f['data'][
                     frame_idxs,
                     row_indices[0]:row_indices[1],
                     col_indices[0]:col_indices[1]
                     ]

        input = self._get_video_clip_for_roi(
            frames=frames,
            fov_shape=self._fov_shape,
            roi=obs.roi
        )

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
                (obs.experiment_id, obs.roi.roi_id)]['agreement']
        else:
            sample_weight = None

        if sample_weight is None:
            return input, target
        else:
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

    def _get_video_clip_for_roi(
            self,
            frames: np.ndarray,
            roi: OphysROI,
            fov_shape: Tuple[int, int],
            normalize_quantiles: Tuple[float, float] = (0.2, 0.99)
    ):
        if len(frames) < self._n_frames * self._limit_to_n_highest_peaks:
            frames = _temporal_pad_frames(
                desired_seq_len=(
                        self._n_frames * self._limit_to_n_highest_peaks),
                frames=frames
            )

        if frames.shape[1:] != self._image_dim:
            frames = _pad_cutout(
                frames=frames,
                roi=roi,
                desired_shape=self._image_dim,
                fov_shape=self._fov_shape,
                pad_mode='symmetric'
            )

        frames = normalize_array(
            array=frames,
            lower_cutoff=np.quantile(frames, normalize_quantiles[0]),
            upper_cutoff=np.quantile(frames, normalize_quantiles[1])
        )

        frames = _draw_mask_outline_on_frames(
            roi=roi,
            cutout_size=self._image_dim[0],
            fov_shape=fov_shape,
            frames=frames
        )
        return frames


def _generate_mask_image(
    fov_shape: Tuple[int, int],
    roi: OphysROI,
    cutout_size: int
) -> np.ndarray:
    """
    Generate mask image from `roi`, cropped to cutout_size X cutout_size

    Parameters
    ----------
    roi
        `OphysROI`

    Returns
    -------
    uint8 np.ndarray with masked region set to 255
    """
    pixel_array = roi.global_pixel_array.transpose()

    mask = np.zeros(fov_shape, dtype=np.uint8)
    mask[pixel_array[0], pixel_array[1]] = 255

    row_indices = get_cutout_indices(
        center_dim=roi.bounding_box_center_y,
        image_dim=fov_shape[0],
        cutout_dim=cutout_size)
    col_indices = get_cutout_indices(
        center_dim=roi.bounding_box_center_x,
        image_dim=fov_shape[0],
        cutout_dim=cutout_size)

    mask = mask[row_indices[0]:row_indices[1], col_indices[0]:col_indices[1]]
    if mask.shape != (cutout_size, cutout_size):
        mask = _pad_cutout(
            frames=mask,
            desired_shape=(cutout_size, cutout_size),
            fov_shape=fov_shape,
            pad_mode='constant',
            roi=roi
        )

    if mask.sum() == 0:
        raise _EmptyMaskException(f'Mask for roi {roi.roi_id} is empty')

    return mask


def _temporal_pad_frames(
    desired_seq_len: int,
    frames: np.ndarray,
) -> np.ndarray:
    """
    Pad the frames so that the len equals desired_seq_len

    Parameters
    ----------
    desired_seq_len
        Desired number of frames
    frames
        Frames
    Returns
    -------
    frames, potentially padded
    """
    n_pad = desired_seq_len - len(frames)
    frames = np.pad(frames, mode='edge',
                    pad_width=((0, n_pad), (0, 0), (0, 0)))
    return frames


def _draw_mask_outline_on_frames(
    roi: OphysROI,
    frames: np.ndarray,
    fov_shape: Tuple[int, int],
    cutout_size: int,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    mask = _generate_mask_image(
        fov_shape=fov_shape,
        roi=roi,
        cutout_size=cutout_size
    )
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    if frames.shape[-1] != 3:
        # Make it into 3 channels, to draw a colored contour on it
        frames = np.stack([frames, frames, frames], axis=-1)

    for frame in frames:
        cv2.drawContours(frame, contours, -1,
                         color=color,
                         thickness=1)
    return frames


def _pad_cutout(
    frames: np.ndarray,
    roi: OphysROI,
    desired_shape: Tuple[int, int],
    fov_shape: Tuple[int, int],
    pad_mode: str
) -> np.ndarray:
    """If the ROI is too close to the edge of the FOV, then we pad in order
    to have frames of the desired shape"""
    row_pad = get_cutout_padding(
        dim_center=roi.bounding_box_center_y,
        image_dim_size=fov_shape[0],
        cutout_dim=desired_shape[0])
    col_pad = get_cutout_padding(
        dim_center=roi.bounding_box_center_x,
        image_dim_size=fov_shape[0],
        cutout_dim=desired_shape[0])

    if len(frames.shape) == 3:
        # Don't pad temporal dimension
        pad_width = ((0, 0), row_pad, col_pad)
    else:
        pad_width = (row_pad, col_pad)
    kwargs = {'constant_values': 0} if pad_mode == 'constant' else {}
    return np.pad(frames,
                  pad_width=pad_width,
                  mode=pad_mode,
                  **kwargs
                  )


def _downsample_frames(
    frames: np.ndarray,
    downsampling_factor: int
) -> np.ndarray:
    """Samples every `downsampling_factor` frame from `frames`"""
    frames = frames[
        np.arange(0,
                  len(frames),
                  downsampling_factor)
    ]
    return frames
