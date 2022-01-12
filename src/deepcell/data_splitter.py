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
                 center_soma=False,
                 target_name='classification_label'):
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
            center_soma
                Try to center the soma. Valid values are "test", "all" or False
            target_name
                The target. Can be "classification_label" or "bounding_box"
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
        self._target_name = target_name

        if center_soma not in ('test', True, False):
            raise ValueError(f'Invalid value for center_soma. Valid '
                             f'values are "test", True, or False')
        self._center_soma = center_soma

    def get_train_test_split(self, test_size):
        full_dataset = RoiDataset(
            model_inputs=self._model_inputs,
            cre_line=self.cre_line,
            mask_out_projections=self.mask_out_projections,
            image_dim=self.image_dim,
            use_correlation_projection=self._use_correlation_projection,
            target=self._target_name,
            include_only_mask=self._target_name == 'bounding_box'
        )
        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size,
                                     random_state=self.seed)
        train_index, test_index = next(
            sss.split(np.zeros(len(full_dataset)),
                      full_dataset.classification_label))

        train_dataset = self._sample_dataset(
            dataset=full_dataset.model_inputs,
            index=train_index,
            transform=self.train_transform,
            center_soma=self._center_soma == 'all')

        test_dataset = self._sample_dataset(
            dataset=full_dataset.model_inputs,
            index=test_index,
            transform=self.test_transform,
            center_soma=True if self._center_soma else False)

        return train_dataset, test_dataset

    def get_cross_val_split(self, train_dataset: RoiDataset, n_splits=5,
                            shuffle=True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                              random_state=self.seed)
        for train_index, test_index in skf.split(
                np.zeros(len(train_dataset)),
                train_dataset.classification_label):
            train = self._sample_dataset(
                dataset=train_dataset.model_inputs,
                index=train_index,
                transform=self.train_transform,
                center_soma=self._center_soma == 'all'
            )
            valid = self._sample_dataset(
                dataset=train_dataset.model_inputs,
                index=test_index,
                transform=self.test_transform,
                center_soma=True if self._center_soma else False)
            yield train, valid

    def _sample_dataset(self, dataset: List[ModelInput],
                        index: List[int],
                        transform: Optional[Transform] = None,
                        center_soma=False) -> \
            RoiDataset:
        """Returns RoiDataset of Artifacts at index

        Args:
            dataset:
                Initial dataset to sample from
            index:
                List of index of artifacts to construct dataset
            transform:
                optional transform to pass to RoiDataset
            center_soma
                Try to center the soma

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
            try_center_soma_in_frame=center_soma,
            target=self._target_name,
            include_only_mask=self._target_name == 'bounding_box'
        )
