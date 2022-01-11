import unittest
from pathlib import Path

import numpy as np
import torch
from deepcell.datasets.model_input import ModelInput
from torch.utils.data import DataLoader
from torchvision import transforms

from deepcell.models import CNN
from deepcell.trainer import Trainer
from deepcell.data_splitter import DataSplitter
from deepcell.hyperparam_tuning import HyperparamTuner, ParamDistribution
from deepcell.datasets.roi_dataset import RoiDataset


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
                                            transform=None)

    expected_ids = [model_inputs[i].roi_id for i in index]
    actual_ids = [x.roi_id for x in dataset.model_inputs]
    assert expected_ids == actual_ids


class Tests(unittest.TestCase):
    @classmethod
    def test_train_test_ids_different(self):
        data_splitter = DataSplitter()
        train, test = data_splitter.get_train_test_split(test_size=.3)
        common_ids = set(train.roi_ids).intersection(test.roi_ids)
        self.assertEqual(len(common_ids), 0)

    def test_train_loss_0_sanity(self):
        train_transform = transforms.ToTensor()
        train = RoiDataset(manifest_path=self.manifest_path, project_name=self.project_name,
                           debug=True)
        cnn = CNN(conv_cfg=[32, 'M', 64, 'M', 128, 'M'], classifier_cfg=[512, 16], dropout_prob=0.0)
        optimizer = lambda: torch.optim.Adam(cnn.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        classifier = Trainer(model=cnn, n_epochs=5, optimizer=optimizer,
                             criterion=criterion, save_path='./saved_models')
        train_loader = DataLoader(dataset=train, batch_size=2)
        train_metrics, _ = classifier.train(train_loader=train_loader)
        self.assertLess(train_metrics.losses[-1], 1e-5)

    def test_cross_validate(self):
        transform = transforms.ToTensor()
        train = RoiDataset(manifest_path=self.manifest_path, project_name=self.project_name, transform=transform)
        cnn = CNN(conv_cfg=[32, 'M', 64, 'M', 128, 'M'], classifier_cfg=[512])
        optimizer = lambda: torch.optim.Adam(cnn.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        classifier = Trainer(model=cnn, train=train, n_epochs=1, optimizer=optimizer,
                             criterion=criterion, save_path='./saved_models')
        data_splitter = DataSplitter(manifest_path=self.manifest_path, project_name=self.project_name,
                                     train_transform=transform, test_transform=transform)
        classifier.cross_validate(data_splitter=data_splitter, n_splits=5)

    def test_hyperparam_tuner(self):
        param_distributions = {
            'optimizer': {
                'optimizer': torch.optim.Adam,
                'params': {
                    'lr': {
                        'distr': (-4, -2),
                        'distr_type': 'LOG_SCALE'
                    }
                }
            }
        }
        param_distributions = ParamDistribution(param_distribution=param_distributions)
        transform = transforms.ToTensor()
        train = RoiDataset(manifest_path=self.manifest_path, project_name=self.project_name, transform=transform)
        data_splitter = DataSplitter(manifest_path=self.manifest_path, project_name=self.project_name,
                                     train_transform=transform, test_transform=transform)
        hp = HyperparamTuner(param_distributions=param_distributions, data_splitter=data_splitter, iters=1)
        res = hp.search(train_dataset=train, n_epochs=1)
        print(res)


if __name__ == '__main__':
    unittest.main()