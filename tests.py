import unittest

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from CNN import CNN
from Classifier import Classifier
from DataSplitter import DataSplitter
from HyperparamTuning import HyperparamTuner, ParamDistribution
from SlcDataset import SlcDataset


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = 's3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020/20201020135214/expert_output/ophys-experts-slc-oct-2020/manifests/output/output.manifest'
        cls.project_name = 'ophys-experts-slc-oct-2020'

    def test_train_test_ids_different(self):
        data_splitter = DataSplitter(manifest_path=self.manifest_path, project_name=self.project_name)
        train, test = data_splitter.get_train_test_split(test_size=.3)
        common_ids = set(train.roi_ids).intersection(test.roi_ids)
        self.assertEqual(len(common_ids), 0)

    def test_train_loss_0_sanity(self):
        train_transform = transforms.ToTensor()
        train = SlcDataset(manifest_path=self.manifest_path, project_name=self.project_name, transform=train_transform,
                           debug=True)
        cnn = CNN(conv_cfg=[32, 'M', 64, 'M', 128, 'M'], classifier_cfg=[512], dropout_prob=0.0)
        optimizer = lambda: torch.optim.Adam(cnn.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        classifier = Classifier(model=cnn, train=train, n_epochs=20, optimizer=optimizer,
                                criterion=criterion, save_path='./saved_models')
        train_loader = DataLoader(dataset=train, batch_size=2)
        train_metrics, _ = classifier.fit(train_loader=train_loader)
        self.assertLess(train_metrics.losses[-1], 1e-5)

    def test_cross_validate(self):
        transform = transforms.ToTensor()
        train = SlcDataset(manifest_path=self.manifest_path, project_name=self.project_name, transform=transform)
        cnn = CNN(conv_cfg=[32, 'M', 64, 'M', 128, 'M'], classifier_cfg=[512])
        optimizer = lambda: torch.optim.Adam(cnn.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        classifier = Classifier(model=cnn, train=train, n_epochs=1, optimizer=optimizer,
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
        train = SlcDataset(manifest_path=self.manifest_path, project_name=self.project_name, transform=transform)
        data_splitter = DataSplitter(manifest_path=self.manifest_path, project_name=self.project_name,
                                     train_transform=transform, test_transform=transform)
        hp = HyperparamTuner(param_distributions=param_distributions, data_splitter=data_splitter, iters=1)
        res = hp.search(train_dataset=train, n_epochs=1)
        print(res)


if __name__ == '__main__':
    unittest.main()