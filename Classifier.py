import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from KfoldDataLoader import KfoldDataLoader

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, model: torch.nn.Module, n_epochs: int, test_loader: DataLoader, optimizer, criterion, save_path,
                 train_loader: DataLoader, kfoldDataLoader: KfoldDataLoader = None, debug=False):
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()
        self.save_path = save_path
        self.kfoldDataLoader = kfoldDataLoader
        self.debug = debug

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')
            
        torch.save(self.model.state_dict(), f'{self.save_path}/model_init.pt')

    def fit(self):
        best_epochs = []
        best_epoch_precisions = []
        best_epoch_recalls = []

        if self.debug:
            data_loaders = [(self.train_loader, None)]
            n_folds = 1
        else:
            data_loaders = self.kfoldDataLoader.run()
            n_folds = self.kfoldDataLoader.n_splits

        logger.info(f'Train evaluate on {n_folds} folds')

        for i, (train_loader, valid_loader) in enumerate(data_loaders):
            logger.info(f'Train/evaluate on fold {i}')

            train_losses, valid_losses, precisions, recalls = self._train(
                n_epochs=self.n_epochs, train_loader=train_loader, valid_loader=valid_loader,
                save_model=False
            )
            best_epoch = valid_losses.argmin()
            best_epochs.append(best_epoch)
            best_epoch_precisions.append(precisions[best_epoch])
            best_epoch_recalls.append(recalls[best_epoch])

            logger.info(f'Best epoch is {best_epoch}')

            # Reset model weights
            logger.info('Resetting model weights')
            state_dict = torch.load(f'{self.save_path}/model_init.pt')
            self.model.load_state_dict(state_dict)

        n_epochs = int(np.median(best_epochs))
        mean_precision = sum(best_epoch_precisions) / len(best_epoch_precisions)
        mean_recall = sum(best_epoch_recalls) / len(best_epoch_recalls)

        logger.info(f'Done train/evaluate for {n_folds} folds')

        logger.info(f'Mean best epoch: {n_epochs}')
        logger.info(f'Mean precision: {mean_precision}')
        logger.info(f'Mean recall: {mean_recall}')

    def _train(self, n_epochs, train_loader: DataLoader, valid_loader: DataLoader = None, save_model=False):
        train_losses = np.zeros(n_epochs)
        valid_losses = np.zeros(n_epochs)
        precisions = np.zeros(n_epochs)
        recalls = np.zeros(n_epochs)

        for epoch in range(n_epochs):
            train_loss = 0.0
            valid_loss = 0.0

            TP = 0
            FP = 0
            FN = 0

            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.float())
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() / len(self.train_loader)

            train_losses[epoch] = train_loss

            if valid_loader:
                self.model.eval()
                for batch_idx, (data, target) in enumerate(valid_loader):
                    # move to GPU
                    if self.use_cuda:
                        data, target = data.cuda(), target.cuda()

                    # update the average validation loss
                    with torch.no_grad():
                        output = self.model(data)
                        output = output.squeeze()
                        loss = self.criterion(output, target.float())
                        valid_loss += loss.item() / len(valid_loader)   # Loss*bs/N = (Sum of all l)/N

                        y_pred = output > .5
                        TP += ((target == 1) & (y_pred == 1)).sum().item()
                        FN += ((target == 1) & (y_pred == 0)).sum().item()
                        FP += ((target == 0) & (y_pred == 1)).sum().item()
                valid_losses[epoch] = valid_loss

                try:
                    precision = TP / (TP + FP)
                except ZeroDivisionError:
                    logger.warning('Precision undefined')
                    precision = 0.0

                try:
                    recall = TP / (TP + FN)
                except ZeroDivisionError:
                    logger.warning('Recall undefined')
                    recall = 0.0

                precisions[epoch] = precision
                recalls[epoch] = recall

            logger.info(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_path}/model.pt')

        return train_losses, valid_losses, precisions, recalls

    def evaluate(self):
        state_dict = torch.load(f'{self.save_path}/model.pt')
        self.model.load_state_dict(state_dict)

        TP = 0
        FP = 0
        FN = 0

        self.model.eval()
        for data, target in self.test_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(data)
                output = output.squeeze()
                y_pred = output > .5
                TP += ((target == 1) & (y_pred == 1)).sum().item()
                FN += ((target == 1) & (y_pred == 0)).sum().item()
                FP += ((target == 0) & (y_pred == 1)).sum().item()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')

