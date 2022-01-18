from typing import List, Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.datasets.visual_behavior_dataset import VisualBehaviorDataset
from deepcell.transform import Transform


class DataSplitter:
    def __init__(self, model_inputs: List[ModelInput], train_transform=None,
                 test_transform=None, seed=None,
                 cre_line=None, exclude_mask=False,
                 mask_out_projections=False, image_dim=(128, 128),
                 use_correlation_projection=False,
                 center_roi_centroid=False,
                 centroid_brightness_quantile=0.8,
                 centroid_use_mask=False):
        """
        Does splitting of data into train/test or train/validation

        Args:
            model_inputs:
                List of model inputs
            train_transform:
                Transforms to apply to training data
            test_transform:
                Transforms to apply to test data
            seed:
                Seed for random shuffling
            cre_line:
                Whether to filter to cre_line
            exclude_mask:
                Whether to exclude the mask from input
            mask_out_projections:
                Whether to mask out the projections using the segmentation mask
            image_dim:
                Input image dimension
            use_correlation_projection:
                Whether to use correlation projection instead of avg
            center_roi_centroid
                See `RoiDataset.center_roi_centroid`.
                Valid values are "test", True or False
                "test" applies centering only at test time, True applies in
                both train and test, and False doesn't apply it
            centroid_brightness_quantile
                See `centroid_brightness_quantile` arg in
                `RoiDataset.centroid_brightness_quantile`
            centroid_use_mask
                See `use_mask` arg in
                `deepcell.datasets.util.calc_roi_centroid`
        """
        self._model_inputs = model_inputs
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed
        self.cre_line = cre_line
        self.exclude_mask = exclude_mask
        self.mask_out_projections = mask_out_projections
        self.image_dim = image_dim
        self._use_correlation_projection = use_correlation_projection
        self._centroid_brightness_quantile = centroid_brightness_quantile
        self._centroid_use_mask = centroid_use_mask

        if center_roi_centroid not in ('test', True, False):
            raise ValueError(f'Invalid value for center_soma. Valid '
                             f'values are "test", True, or False')
        self._center_roi_centroid = center_roi_centroid

    def get_train_test_split(self, test_size):
        full_dataset = RoiDataset(
            model_inputs=self._model_inputs,
            cre_line=self.cre_line,
            mask_out_projections=self.mask_out_projections,
            image_dim=self.image_dim,
            use_correlation_projection=self._use_correlation_projection
        )
        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size,
                                     random_state=self.seed)
        train_index, test_index = next(sss.split(np.zeros(len(full_dataset)),
                                                 full_dataset.y))

        test_center_roi_centroid = \
            self._center_roi_centroid is True or \
            self._center_roi_centroid == 'test'

        train_dataset = self._sample_dataset(
            dataset=full_dataset.model_inputs,
            index=train_index,
            transform=self.train_transform,
            center_roi_centroid=self._center_roi_centroid is True)
        
        test_dataset = self._sample_dataset(
            dataset=full_dataset.model_inputs,
            index=test_index,
            transform=self.test_transform,
            center_roi_centroid=test_center_roi_centroid)

        return train_dataset, test_dataset

    def get_cross_val_split(self, train_dataset: RoiDataset, n_splits=5,
                            shuffle=True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                              random_state=self.seed)
        for train_index, test_index in skf.split(np.zeros(len(train_dataset)),
                                                 train_dataset.y):
            train = self._sample_dataset(
                dataset=train_dataset.model_inputs,
                index=train_index,
                transform=self.train_transform,
                center_roi_centroid=self._center_roi_centroid == 'all'
            )
            valid = self._sample_dataset(
                dataset=train_dataset.model_inputs,
                index=test_index,
                transform=self.test_transform,
                center_roi_centroid=True if self._center_roi_centroid else False)
            yield train, valid

    def _sample_dataset(self, dataset: List[ModelInput],
                        index: List[int],
                        transform: Optional[Transform] = None,
                        center_roi_centroid=False) -> \
            RoiDataset:
        """Returns RoiDataset of Artifacts at index

        Args:
            dataset:
                Initial dataset to sample from
            index:
                List of index of artifacts to construct dataset
            transform:
                optional transform to pass to RoiDataset
            center_roi_centroid
                See `RoiDataset.center_roi_centroid`

        Returns
            RoiDataset
        """
        artifacts = [dataset[i] for i in index]
        return RoiDataset(
            model_inputs=artifacts,
            transform=transform,
            exclude_mask=self.exclude_mask,
            mask_out_projections=self.mask_out_projections,
            image_dim=self.image_dim,
            use_correlation_projection=self._use_correlation_projection,
            center_roi_centroid=center_roi_centroid,
            centroid_brightness_quantile=self._centroid_brightness_quantile,
            centroid_use_mask=self._centroid_use_mask
        )
