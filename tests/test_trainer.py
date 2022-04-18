import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader

from deepcell.callbacks.early_stopping import EarlyStopping
from deepcell.trainer import Trainer


class TestTrainer:
    @classmethod
    def setup_class(cls):
        inputs = torch.zeros((2, 3, 224, 224))
        for i in range(2):
            input = np.random.randn(224, 224, 3)
            inputs[i] = torchvision.transforms.ToTensor()(input)
        targets = torch.tensor([0, 1])
        dataset = TensorDataset(inputs, targets)
        cls.train_loader = DataLoader(dataset, batch_size=2)

    def setup_method(self):
        net = torchvision.models.resnet18(pretrained=True, progress=False)
        net.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 1)
        )
        self.net = net

    @pytest.mark.parametrize('early_stopping_params',
                             [('f1', True), ('loss', False)])
    @pytest.mark.parametrize('patience', [0, 3])
    def test_early_stopping(self, early_stopping_params, patience):
        """Tests that early stopping is triggered on tiny dataset before
        n_epochs
        """
        metric, larger_better = early_stopping_params
        with tempfile.TemporaryDirectory() as f:
            trainer = Trainer(
                model=self.net,
                n_epochs=1000,
                optimizer=torch.optim.Adam(self.net.parameters(), lr=1e-2),
                criterion=torch.nn.BCEWithLogitsLoss(),
                save_path=f,
                callbacks=[
                    EarlyStopping(patience=patience, best_metric=metric,
                                  metric_larger_is_better=larger_better)
                ]
            )

            trainer.train(train_loader=self.train_loader,
                          valid_loader=self.train_loader)
            assert trainer.early_stopping_callback.best_epoch < \
                   trainer.n_epochs
            func = np.argmax if larger_better else np.argmin
            be = trainer.early_stopping_callback.best_epoch
            assert func(
                trainer._callback_metrics[f'val_{metric}']
                [:be + patience + 1]) == be

    def test_train_loss_sanity(self):
        """Tests that when we train on a dummy dataset of 2 examples with 2
        different classes, that the model is able to overfit and achieve a
        loss close to 0"""
        with tempfile.TemporaryDirectory() as f:
            trainer = Trainer(
                model=self.net,
                n_epochs=30,
                optimizer=torch.optim.Adam(self.net.parameters(), lr=1e-4),
                criterion=torch.nn.BCEWithLogitsLoss(),
                save_path=f
            )

            trainer.train(train_loader=self.train_loader,
                          valid_loader=self.train_loader)

            checkpoint = torch.load(Path(f) / 'model.pt')
            assert checkpoint['performance']['loss'][-1] < 1e-3

    def test_continue_training(self):
        """Tests continuing training"""
        with tempfile.TemporaryDirectory() as f:
            trainer = Trainer(
                model=self.net,
                n_epochs=1000,
                optimizer=torch.optim.Adam(self.net.parameters(), lr=1e-4),
                criterion=torch.nn.BCEWithLogitsLoss(),
                save_path=f,
                callbacks=[
                    EarlyStopping(patience=0, best_metric='f1')
                ]
            )

            trainer.train(train_loader=self.train_loader,
                          valid_loader=self.train_loader)

            trainer_continue = Trainer(
                model=self.net,
                model_load_path=f,
                n_epochs=1000,
                optimizer=torch.optim.Adam(self.net.parameters(), lr=1e-4),
                criterion=torch.nn.BCEWithLogitsLoss(),
                save_path=f,
                callbacks=[
                    EarlyStopping(patience=0, best_metric='f1')
                ]
            )
            trainer_continue.train(train_loader=self.train_loader,
                                   valid_loader=self.train_loader)
            assert trainer_continue.early_stopping_callback.best_epoch >= \
                   trainer.early_stopping_callback.best_epoch

            for k in trainer._callback_metrics:
                assert (trainer._callback_metrics[k]
                        [:trainer.early_stopping_callback.best_epoch + 1] ==
                        (trainer_continue._callback_metrics[k]
                        [:trainer.early_stopping_callback.best_epoch + 1]))
