from pathlib import Path

import numpy as np

from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.exp_metadata import ExperimentMetadata


def test_ids_different():
    """Tests that the following ids are different:
    1. Train/test
    2. Train CV split / Val CV split
    3. Train CV split / Test
    4. Val CV split / Test"""
    model_inputs = [ModelInput(roi_id=f'{i}',
                               experiment_id=f'{i%10}',
                               mask_path=Path('foo'),
                               max_projection_path=Path('foo'),
                               avg_projection_path=Path('foo')
                               ) for i in range(5000)]
    exp_inputs = [ExperimentMetadata(experiment_id=idx,
                                     imaging_depth=idx % 2,
                                     equipment='2P',
                                     problem_experiment=False)
                  for idx in range(10)]

    data_splitter = \
        DataSplitter(model_inputs=model_inputs,
                     experiment_metadatas=exp_inputs,
                     seed=1234)
    train, test = data_splitter.get_train_test_split(test_size=.3,
                                                     full_dataset=model_inputs,
                                                     exp_metas=exp_inputs,
                                                     n_depth_bins=2)
    assert len(set([x.roi_id for x in train.model_inputs]).intersection(
        [x.roi_id for x in test.model_inputs])) == 0
    for train, val in data_splitter.get_cross_val_split(train_dataset=train):
        assert len(set([x.roi_id for x in train.model_inputs]).intersection(
            [x.roi_id for x in val.model_inputs])) == 0
        assert len(set([x.roi_id for x in train.model_inputs]).intersection(
            [x.roi_id for x in test.model_inputs])) == 0
        assert len(set([x.roi_id for x in val.model_inputs]).intersection(
            [x.roi_id for x in test.model_inputs])) == 0


def test_dataset_is_shuffled():
    """Tests that when train/test split is performed, the records are
    shuffled"""
    model_inputs = [ModelInput(roi_id=f'{i}',
                               experiment_id=f'{i%10}',
                               mask_path=Path('foo'),
                               max_projection_path=Path('foo'),
                               avg_projection_path=Path('foo')
                               ) for i in range(5000)]
    exp_inputs = [ExperimentMetadata(experiment_id=idx,
                                     imaging_depth=idx % 2,
                                     equipment='2P',
                                     problem_experiment=False)
                  for idx in range(10)]
    data_splitter = \
        DataSplitter(model_inputs=model_inputs,
                     experiment_metadatas=exp_inputs,
                     seed=1234)

    index = np.random.choice(range(len(model_inputs)), size=100, replace=False)
    dataset = data_splitter._sample_dataset(dataset=model_inputs, index=index,
                                            transform=None, is_train=True)

    expected_ids = [model_inputs[i].roi_id for i in index]
    actual_ids = [x.roi_id for x in dataset.model_inputs]
    assert expected_ids == actual_ids


def test_experiment_binning():
    """Tests that the code for binning experiment data.
    """
    # Just need model inputs for the class init.
    model_inputs = [ModelInput(roi_id=f'{i}',
                               experiment_id=f'{i%10}',
                               mask_path=Path('foo'),
                               max_projection_path=Path('foo'),
                               avg_projection_path=Path('foo')
                               ) for i in range(5)]
    # Create 2 sets of experiments. 2P at various depths identified as 
    # special or "problem" experiments and a set of special experiments.
    n_exp = 4
    n_depth_bins = 2
    exp_inputs = [ExperimentMetadata(experiment_id=idx,
                                     imaging_depth=idx % 4,
                                     equipment='2P',
                                     problem_experiment=False)
                  for idx in range(n_exp)]
    exp_inputs.extend([ExperimentMetadata(experiment_id=idx,
                                          imaging_depth=idx,
                                          equipment='2P',
                                          problem_experiment=True)
                       for idx in range(n_exp, 2 * n_exp)])

    data_splitter = \
        DataSplitter(model_inputs=model_inputs,
                     experiment_metadatas=exp_inputs,
                     seed=1234)
    exp_bin_ids, _ = data_splitter._get_experiment_groups_for_stratified_split(
        experiment_metadatas=exp_inputs,
        n_depth_bins=n_depth_bins)
    assert (exp_bin_ids == 0).sum() == n_exp
    assert (exp_bin_ids == 1).sum() == n_exp / n_depth_bins
    assert (exp_bin_ids == 2).sum() == n_exp / n_depth_bins


def test_convert_exp_index_to_roi_index():
    """Test that the mapping from selected experiment to ROIs is working as
    intended.
    """
    model_inputs = [ModelInput(roi_id=f'{i}',
                               experiment_id=f'{i%10}',
                               mask_path=Path('foo'),
                               max_projection_path=Path('foo'),
                               avg_projection_path=Path('foo')
                               ) for i in range(10)]
    # Just need some experiment inputs here.
    exp_inputs = [ExperimentMetadata(experiment_id=idx,
                                     imaging_depth=idx % 4,
                                     equipment='2P',
                                     problem_experiment=False)
                  for idx in range(10)]
    expected_exps = np.array([1, 2])
    data_splitter = \
        DataSplitter(model_inputs=model_inputs,
                     experiment_metadatas=exp_inputs,
                     seed=1234)
    rois_idxs = data_splitter._convert_exp_index_to_roi_index(
        exp_ids=np.arange(10, dtype=int),
        exp_indices=expected_exps,
        full_dataset=model_inputs)
    assert len(rois_idxs) == len(expected_exps)
    assert rois_idxs[0] == 1
    assert rois_idxs[1] == 2
