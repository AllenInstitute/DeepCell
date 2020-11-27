import logging
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader

from Metrics import Metrics, TrainingMetrics
from SlcDataset import SlcDataset

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, model: torch.nn.Module, train: SlcDataset, n_epochs: int, optimizer,
                 criterion, save_path, scheduler=None, scheduler_step_after_batch=False, debug=False):
        self.n_epochs = n_epochs
        self.model = model
        self.train_dataset = train
        self.optimizer_constructor = optimizer
        self.scheduler_contructor = scheduler
        self.optimizer = optimizer()
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None
        self.scheduler_step_after_batch = scheduler_step_after_batch
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()
        self.save_path = save_path
        self.debug = debug

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')

        if self.use_cuda:
            self.model.cuda()

        torch.save(self.model.state_dict(), f'{self.save_path}/model_init.pt')

    def cross_validate(self, data_splitter, n_splits=5, shuffle=True, batch_size=64, log_after_each_epoch=True):
        if self.debug:
            datasets = [(self.train_dataset, None)]
            n_splits = 1
        else:
            datasets = data_splitter.get_cross_val_split(train_dataset=self.train_dataset, n_splits=n_splits,
                                                         shuffle=shuffle)
        train_losses = np.zeros((n_splits, self.n_epochs))
        val_losses = np.zeros((n_splits, self.n_epochs))
        precisions = np.zeros((n_splits, self.n_epochs))
        recalls = np.zeros((n_splits, self.n_epochs))
        train_f1s = np.zeros((n_splits, self.n_epochs))
        val_f1s = np.zeros((n_splits, self.n_epochs))
        best_epochs = np.zeros(n_splits, dtype=np.int)

        logger.info(f'Train evaluate on {n_splits} folds')

        for i, (train, valid) in enumerate(datasets):
            train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size)
            valid_loader = DataLoader(dataset=valid, shuffle=False, batch_size=batch_size)

            logger.info(f'Train/evaluate on fold {i}')

            train_metrics, val_metrics = self.train(
                train_loader=train_loader, valid_loader=valid_loader, save_model=False,
                log_after_each_epoch=log_after_each_epoch
            )
            best_epochs[i] = val_metrics.losses.argmin()
            precisions[i] = val_metrics.precisions
            recalls[i] = val_metrics.recalls
            val_f1s[i] = val_metrics.f1s

            train_f1s[i] = train_metrics.f1s

            train_losses[i] = train_metrics.losses
            val_losses[i] = val_metrics.losses

            # Reset
            logger.info('Resetting model')
            self._reset()
        res = {
            'all_val_precision': precisions.mean(axis=0),
            'all_val_recall': recalls.mean(axis=0),
            'all_val_f1': val_f1s.mean(axis=0),
            'all_train_f1': train_f1s.mean(axis=0),
            'train_losses': train_losses.mean(axis=0),
            'val_losses': val_losses.mean(axis=0),
            'precision_mean': precisions[:, -1].mean(),
            'precision_std': precisions[:, -1].std(),
            'recall_mean': recalls[:, -1].mean(),
            'recall_std': recalls[:, -1].std(),
            'f1_mean': val_f1s[:, -1].mean(),
            'f1_std': val_f1s[:, -1].std(),
        }

        logger.info(f'Done train/evaluate for {n_splits} folds')

        logger.info({
            'precision_mean': res['precision_mean'],
            'precision_std': res['precision_std'],
            'recall_mean': res['recall_mean'],
            'recall_std': res['recall_std'],
            'f1_mean': res['f1_mean'],
            'f1_std': res['f1_std']
        })

        return res

    def train(self, train_loader: DataLoader, valid_loader: DataLoader = None, save_model=False,
              log_after_each_epoch=True):
        all_train_metrics = TrainingMetrics(n_epochs=self.n_epochs)
        all_val_metrics = TrainingMetrics(n_epochs=self.n_epochs)

        for epoch in range(self.n_epochs):
            epoch_train_metrics = Metrics()
            epoch_val_metrics = Metrics()

            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()
                output = self.model(data)
                output = output.squeeze()
                loss = self.criterion(output, target.float())
                loss.backward()
                self.optimizer.step()
                if self.scheduler_step_after_batch:
                    if self.scheduler is not None:
                        self.scheduler.step()

                epoch_train_metrics.update_loss(loss=loss.item(), num_batches=len(train_loader))
                epoch_train_metrics.update_accuracies(y_true=target, y_out=output)

            all_train_metrics.update(epoch=epoch,
                                     loss=epoch_train_metrics.loss,
                                     precision=epoch_train_metrics.precision,
                                     recall=epoch_train_metrics.recall,
                                     f1=epoch_train_metrics.F1)

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

                        epoch_val_metrics.update_loss(loss=loss.item(), num_batches=len(valid_loader))
                        epoch_val_metrics.update_accuracies(y_true=target, y_out=output)

            if not self.scheduler_step_after_batch:
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(epoch_val_metrics.F1)
                    else:
                        self.scheduler.step()

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       precision=epoch_val_metrics.precision,
                                       recall=epoch_val_metrics.recall,
                                       f1=epoch_val_metrics.F1)

            if log_after_each_epoch:
                logger.info(f'Epoch: {epoch + 1} \tTrain Loss: {epoch_train_metrics.loss:.6f} '
                            f'\tTrain F1: {epoch_train_metrics.F1:.6f}'
                            f'\tVal F1: {epoch_val_metrics.F1:.6f}')

        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_path}/model.pt')

        return all_train_metrics, all_val_metrics

    def test(self, test_loader: DataLoader):
        state_dict = torch.load(f'{self.save_path}/model.pt')
        self.model.load_state_dict(state_dict)

        metrics = Metrics()

        self.model.eval()
        for data, target in test_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(data)
                output = output.squeeze()
                metrics.update_accuracies(y_true=target, y_out=output)

        return metrics

    def _reset(self):
        # reset model weights
        state_dict = torch.load(f'{self.save_path}/model_init.pt')
        self.model.load_state_dict(state_dict)

        # reset optimizer
        self.optimizer = self.optimizer_constructor()

        # reset scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler_contructor(self.optimizer)

