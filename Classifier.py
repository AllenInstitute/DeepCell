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
    def __init__(self, model: torch.nn.Module, n_epochs: int, optimizer,
                 criterion, save_path, scheduler=None, scheduler_step_after_batch=False, debug=False):
        self.n_epochs = n_epochs
        self.model = model
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

    def train(self, train_loader: DataLoader, valid_loader: DataLoader = None,
              log_after_each_epoch=True, early_stopping_count=30):
        all_train_metrics = TrainingMetrics(n_epochs=self.n_epochs)
        all_val_metrics = TrainingMetrics(n_epochs=self.n_epochs)

        best_epoch_loss = float('inf')
        best_epoch = 0
        time_since_best_epoch = 0

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

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       precision=epoch_val_metrics.precision,
                                       recall=epoch_val_metrics.recall,
                                       f1=epoch_val_metrics.F1)

                if epoch_val_metrics.loss < best_epoch_loss:
                    torch.save(self.model.state_dict(), f'{self.save_path}/model.pt')
                    best_epoch = epoch
                    best_epoch_loss = epoch_val_metrics.loss
                    time_since_best_epoch = 0
                else:
                    time_since_best_epoch += 1
                    if time_since_best_epoch > early_stopping_count:
                        logger.info('Stopping due to early stopping')
                        all_train_metrics.truncate_to_epoch(epoch=best_epoch)
                        all_val_metrics.truncate_to_epoch(epoch=best_epoch)
                        return all_train_metrics, all_val_metrics

            if not self.scheduler_step_after_batch:
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(epoch_val_metrics.F1)
                    else:
                        self.scheduler.step()

            if log_after_each_epoch:
                logger.info(f'Epoch: {epoch + 1} \tTrain Loss: {epoch_train_metrics.loss:.6f} '
                            f'\tTrain F1: {epoch_train_metrics.F1:.6f}'
                            f'\tVal F1: {epoch_val_metrics.F1:.6f}')

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

