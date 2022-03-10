from pathlib import Path

import numpy as np

from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput


def test_ids_different():
    """Tests that the following ids are different:
    1. Train/test
    2. Train CV split / Val CV split
    3. Train CV split / Test
    4. Val CV split / Test"""
    model_inputs = [ModelInput(roi_id=f'{i}',
                               experiment_id='foo',
                               mask_path=Path('foo'),
                               max_projection_path=Path('foo'),
                               avg_projection_path=Path('foo')
                               ) for i in range(5000)]
    data_splitter = \
        DataSplitter(model_inputs=model_inputs, seed=1234)
    train, test = data_splitter.get_train_test_split(test_size=.3)
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
                               experiment_id='foo',
                               mask_path=Path('foo'),
                               max_projection_path=Path('foo'),
                               avg_projection_path=Path('foo')
                               ) for i in range(5000)]
    data_splitter = \
        DataSplitter(model_inputs=model_inputs, seed=1234)

    index = np.random.choice(range(len(model_inputs)), size=100, replace=False)
    dataset = data_splitter._sample_dataset(dataset=model_inputs, index=index,
                                            transform=None, is_train=True)

    expected_ids = [model_inputs[i].roi_id for i in index]
    actual_ids = [x.roi_id for x in dataset.model_inputs]
    assert expected_ids == actual_ids
