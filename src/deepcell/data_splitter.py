from typing import List, Optional, Tuple, Iterator

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.datasets.exp_metadata import ExperimentMetadata
from deepcell.transform import Transform


class DataSplitter:
    def __init__(self,
                 model_inputs: List[ModelInput],
                 experiment_metadatas: List[ExperimentMetadata],
                 train_transform=None,
                 test_transform=None, seed=None,
                 cre_line=None, exclude_mask=False,
                 mask_out_projections=False, image_dim=(128, 128),
                 use_correlation_projection=True,
                 center_roi_centroid=False,
                 centroid_brightness_quantile=0.8,
                 centroid_use_mask=False):
        """
        Does splitting of data into train/test or train/validation

        Args:
            model_inputs:
                List of model inputs
            experiment_metadatas:
                List of experiment metadata.
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
        self._exp_metas = experiment_metadatas
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
            raise ValueError('Invalid value for center_soma. Valid '
                             'values are "test", True, or False')
        self._center_roi_centroid = center_roi_centroid

    @staticmethod
    def _get_experiment_groups_for_stratified_split(
            experiment_metadatas: List[ExperimentMetadata],
            n_depth_bins: int):
        """Bin up input experiment data into bins of depth and "problem"
        experiments.

        Parameters
        ---------
        experiment_metadatas : List[ExperimentMetadata]
            Set of experiment metadatas to stratify.
        n_depth_bins
            Number of of bins to split by experiment depth. Bins are
            selected to have an equal number of experiments in each bin.

        Returns
        -------
        exp_bin_ids : numpy.ndarray
           Bin assigned to each experiment id.
        """
        # Create the arrays we will use to bin. -1 for a bin_id denotes
        # an experiment that has not been assigned to a bit yet.
        exp_bin_ids = np.full(len(experiment_metadatas), -1, dtype=int)
        depths = np.empty(len(experiment_metadatas), dtype=int)

        for idx, exp_meta in enumerate(experiment_metadatas):
            # Check for any "special" experiments that we want to partition
            # separately. This can be anything from identifying a set that have
            # problematic segmentation or have some other special attribute
            # we would like to sample fairly in the train/test split.
            if exp_meta['problem_experiment']:
                exp_bin_ids[idx] = 0
            depths[idx] = exp_meta['imaging_depth']

        # Create linear spaced bins from min->max depth. Bin indexes start
        # from 1 to n_depth_bins.
        bin_edges = np.linspace(depths[exp_bin_ids < 0].min() - 1e-8,
                                depths[exp_bin_ids < 0].max() + 1e-8,
                                num=n_depth_bins + 1)
        bin_ids = np.digitize(depths[exp_bin_ids < 0], bin_edges)
        exp_bin_ids[exp_bin_ids < 0] = bin_ids

        return exp_bin_ids, bin_edges

    @staticmethod
    def _convert_exp_index_to_roi_index(
            exp_ids: np.ndarray,
            exp_indices: List[int],
            full_dataset: List[ModelInput]
    ) -> np.ndarray:
        """Given a set of indices in the the experiment list, find all
        ROIs that are in the experiments.

        Parameters
        ----------
        exp_ids : np.ndarray
            Full set of experiment ids
        exp_indices : List[int]
            Indexes of selecting experiments in ``exp_ids`` array.
        full_dataset : List[ModelInput]
            Set of ROIs for all experiments.

        Returns
        -------
        roi_indexes : np.ndarray
            Subset of ROIs that are within the selected experiments.
        """
        selected_experiments = exp_ids[exp_indices]
        roi_indexes = []
        for idx, data in enumerate(full_dataset):
            if np.isin(int(data.experiment_id), selected_experiments):
                roi_indexes.append(idx)
        roi_indexes = np.array(roi_indexes)
        return roi_indexes

    def get_train_test_split(self,
                             test_size: float,
                             full_dataset: List[ModelInput],
                             exp_metas: List[ExperimentMetadata],
                             n_depth_bins: int = 5):
        """Create a train/test split ofROIs by experiment preserving the
        fraction in bins of depth, experiment type (2P vs 3P), and identified
        problem/special experiments.

        Parameters
        ----------
        test_size : float
            Percentage of data to reserve for test sample.
        full_dataset: List[ModelInput]
            Full set of rois.
        exp_metas : RoiDataset
            Set of experiment metadatas to stratify.
        n_depth_bins
            Number of of bins to split by experiment depth. Bins are
            selected to have an equal number of experiments in each bin.

        Returns
        -------
        train_test_indices : Tuple[RoiDataset, RoiDataset]
            Datasets split by train/test.
        """
        exp_ids = np.array([int(exp_meta['experiment_id'])
                            for exp_meta in exp_metas], dtype='uint64')
        exp_bin_ids, _ = self._get_experiment_groups_for_stratified_split(
            experiment_metadatas=exp_metas, n_depth_bins=n_depth_bins)
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=test_size,
                                     random_state=self.seed)
        train_index, test_index = next(sss.split(exp_bin_ids,
                                                 exp_bin_ids))

        train_dataset = self._sample_dataset(
            dataset=full_dataset,
            index=self._convert_exp_index_to_roi_index(
                exp_ids,
                train_index,
                full_dataset),
            transform=self.train_transform,
            is_train=True
        )

        test_dataset = self._sample_dataset(
            dataset=full_dataset,
            index=self._convert_exp_index_to_roi_index(
                exp_ids,
                test_index,
                full_dataset),
            transform=self.test_transform,
            is_train=False)

        return train_dataset, test_dataset

    def get_cross_val_split(self, train_dataset: RoiDataset, n_splits=5,
                            shuffle=True):
        for train_index, test_index in self.get_cross_val_split_idxs(
                model_inputs=train_dataset.model_inputs,
                n_splits=n_splits,
                shuffle=shuffle):
            train = self._sample_dataset(
                dataset=train_dataset.model_inputs,
                index=train_index,
                transform=self.train_transform,
                is_train=True
            )
            valid = self._sample_dataset(
                dataset=train_dataset.model_inputs,
                index=test_index,
                transform=self.test_transform,
                is_train=False)
            yield train, valid

    @staticmethod
    def get_cross_val_split_idxs(
            model_inputs: List[ModelInput],
            n_splits=5,
            shuffle=True,
            seed=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Gets the cross validation split indices
        Parameters
        ----------
        model_inputs: List of ModelInput to split
        n_splits: total number of cross validation splits
        shuffle: whether to shuffle before performing the split
        seed: seed for reproducibility of shuffling
        Returns
        -------
        yields Tuple of np.ndarray for train and test indices
        """
        n = len(model_inputs)
        y = RoiDataset.get_numeric_labels(model_inputs=model_inputs)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                              random_state=seed)
        for train_index, test_index in skf.split(np.zeros(n), y):
            yield train_index, test_index

    def _sample_dataset(self, dataset: List[ModelInput],
                        index: np.ndarray,
                        is_train: bool,
                        transform: Optional[Transform] = None) -> \
            RoiDataset:
        """Returns RoiDataset of Artifacts at index
        Args:
            dataset:
                Initial dataset to sample from
            index:
                Array of index of artifacts to construct dataset
            is_train
                Whether this is a train dataset or test dataset
            transform:
                optional transform to pass to RoiDataset
        Returns
            RoiDataset
        """
        if is_train:
            center_roi_centroid = self._center_roi_centroid is True
        else:
            center_roi_centroid = \
                self._center_roi_centroid is True or \
                self._center_roi_centroid == 'test'

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
