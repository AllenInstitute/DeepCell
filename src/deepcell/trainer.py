import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Callable

import torch
from torch.utils.data import DataLoader

from deepcell.data_splitter import DataSplitter
from deepcell.metrics import Metrics, TrainingMetrics, CVMetrics
from deepcell.datasets.roi_dataset import RoiDataset

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            n_epochs: int,
            optimizer: Callable[..., torch.optim.Adam],
            criterion: torch.nn.BCEWithLogitsLoss,
            save_path: Union[str, Path],
            scheduler: Optional[
                Callable[[torch.optim.Adam],
                         torch.optim.lr_scheduler.ReduceLROnPlateau]] = None,
            scheduler_step_after_batch=False,
            early_stopping=30,
            model_load_path: Optional[Union[str, Path]] = None):
        """
        The driver for the training and evaluation loop
        Args:
            model:
                torch.nn.Module
            n_epochs:
                Number of epochs to train for
            optimizer:
                function which returns instantiation of Adam optimizer
            criterion:
                loss function
            save_path:
                Where to save checkpoints
            scheduler:
                optional function which returns learning rate scheduler
            scheduler_step_after_batch:
                Whether the scheduler steps after each batch or epoch
            early_stopping:
                Number of epochs to activate early stopping
            model_load_path:
                Path to load a pretrained model. Activates continuation of
                training
        """
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer_constructor = optimizer
        self.scheduler_contructor = scheduler
        self.optimizer = optimizer()
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None
        self.scheduler_step_after_batch = scheduler_step_after_batch
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()
        self.early_stopping = early_stopping
        self.model_load_path = model_load_path
        self.save_path = save_path
        self._current_best_state_dicts = {}

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')

        if self.use_cuda:
            self.model = self.model.cuda()

        torch.save({
            'state_dict': self.model.state_dict()
        }, f'{self.save_path}/model_init.pt')

    def cross_validate(self, train_dataset: RoiDataset, data_splitter: DataSplitter, batch_size=64, sampler=None,
                       n_splits=5, log_after_each_epoch=True):
        cv_metrics = CVMetrics(n_splits=n_splits)

        for i, (train, valid) in enumerate(data_splitter.get_cross_val_split(train_dataset=train_dataset,
                                                                             n_splits=n_splits)):
            logger.info(f'=========')
            logger.info(f'Fold {i}')
            logger.info(f'=========')

            if self.model_load_path is not None:
                train_metrics, val_metrics = self._load_pretrained_model(
                    checkpoint_path=Path(
                        self.model_load_path) / f'{i}_model.pt')
            else:
                train_metrics = TrainingMetrics(n_epochs=self.n_epochs)
                val_metrics = TrainingMetrics(n_epochs=self.n_epochs)

            train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size, sampler=sampler)
            valid_loader = DataLoader(dataset=valid, shuffle=False, batch_size=batch_size)
            train_metrics, valid_metrics = self.train(
                train_loader=train_loader, valid_loader=valid_loader,
                eval_fold=i,
                log_after_each_epoch=log_after_each_epoch,
                train_metrics=train_metrics, val_metrics=val_metrics)

            cv_metrics.update(train_metrics=train_metrics, valid_metrics=valid_metrics)

            self._reset()

        return cv_metrics

    def train(self, train_loader: DataLoader, eval_fold=None, valid_loader: DataLoader = None,
              log_after_each_epoch=True,
              train_metrics: Optional[TrainingMetrics] = None,
              val_metrics: Optional[TrainingMetrics] = None):
        if train_metrics is not None:
            all_train_metrics = train_metrics
        else:
            all_train_metrics = TrainingMetrics(n_epochs=self.n_epochs,
                                                best_metric='loss',
                                                metric_larger_is_better=False)

        if val_metrics is not None:
            all_val_metrics = val_metrics
        else:
            all_val_metrics = TrainingMetrics(n_epochs=self.n_epochs,
                                              best_metric='loss',
                                              metric_larger_is_better=False)

        time_since_best_epoch = 0

        for epoch in range(val_metrics.best_epoch + 1, self.n_epochs):
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

                epoch_train_metrics.update_loss(loss=loss.item(), num_batches=len(train_loader))
                epoch_train_metrics.update_outputs(y_true=target, y_out=output)

            all_train_metrics.update(epoch=epoch,
                                     loss=epoch_train_metrics.loss,
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
                        epoch_val_metrics.update_outputs(y_true=target, y_out=output)

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       f1=epoch_val_metrics.F1)

                if all_val_metrics.best_epoch == epoch:
                    scheduler_state_dict = self.scheduler.state_dict() if \
                        self.scheduler is not None else None
                    self._current_best_state_dicts = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': scheduler_state_dict
                    }
                    time_since_best_epoch = 0
                else:
                    time_since_best_epoch += 1
                    if time_since_best_epoch > self.early_stopping:
                        logger.info('Stopping due to early stopping')
                        self._save_model_and_performance(
                            eval_fold=eval_fold,
                            all_train_metrics=all_train_metrics,
                            all_val_metrics=all_val_metrics, epoch=epoch)
                        return all_train_metrics, all_val_metrics

            if not self.scheduler_step_after_batch:
                if self.scheduler is not None:
                    self.scheduler.step(epoch_val_metrics.loss)

            if log_after_each_epoch:
                logger.info(f'Epoch: {epoch + 1} \t'
                            f'Train F1: {epoch_train_metrics.F1:.6f} \t'
                            f'Val F1: {epoch_val_metrics.F1:.6f}\t'
                            f'Val Loss: {epoch_val_metrics.loss}'
                            )
        self._save_model_and_performance(eval_fold=eval_fold,
                                         all_train_metrics=all_train_metrics,
                                         all_val_metrics=all_val_metrics,
                                         epoch=self.n_epochs)

        return all_train_metrics, all_val_metrics

    def _reset(self):
        # reset model weights
        x = torch.load(f'{self.save_path}/model_init.pt')
        self.model.load_state_dict(x['state_dict'])

        # reset optimizer
        self.optimizer = self.optimizer_constructor()

        # reset scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler_contructor(self.optimizer)

    def _save_model_and_performance(self, eval_fold: int,
                                    all_train_metrics: TrainingMetrics,
                                    all_val_metrics: TrainingMetrics,
                                    epoch: int):
        """Writes model weights and training performance to disk"""
        torch.save({
            'state_dict': self._current_best_state_dicts['model'],
            'optimizer': self._current_best_state_dicts['optimizer'],
            'scheduler': self._current_best_state_dicts['scheduler'],
            'performance': {
                'train': all_train_metrics.to_dict(
                    to_epoch=epoch, best_epoch=all_train_metrics.best_epoch),
                'val': all_val_metrics.to_dict(
                    to_epoch=epoch, best_epoch=all_val_metrics.best_epoch)
            }
        }, f'{self.save_path}/{eval_fold}_model.pt')

    def _load_pretrained_model(self, checkpoint_path: Path) -> \
            Tuple[TrainingMetrics, TrainingMetrics]:
        """Loads a pretrained model to continue training

        Args:
            checkpoint_path: checkpoint path of pretrained model

        Returns:
            Tuple of training metrics from previous training run
        """
        x = torch.load(checkpoint_path)
        self.model.load_state_dict(x['state_dict'])
        self.optimizer.load_state_dict(x['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(x['scheduler'])
        train_metrics = TrainingMetrics(
            n_epochs=self.n_epochs,
            losses=x['performance']['train']['losses'],
            f1s=x['performance']['train']['f1s'],
            best_epoch=x['performance']['train']['best_epoch'],
            best_metric=x['performance']['train']['best_metric'],
            best_metric_value=x['performance']['train'][
                'best_metric_value'],
            metric_larger_is_better=x['performance']['train'][
                'metric_larger_is_better']
        )
        val_metrics = TrainingMetrics(
            n_epochs=self.n_epochs,
            losses=x['performance']['val']['losses'],
            f1s=x['performance']['val']['f1s'],
            best_epoch=x['performance']['val']['best_epoch'],
            best_metric=x['performance']['val']['best_metric'],
            best_metric_value=x['performance']['val'][
                'best_metric_value'],
            metric_larger_is_better=x['performance']['val'][
                'metric_larger_is_better']
        )

        self._current_best_state_dicts = {
            'model': x['state_dict'],
            'optimizer': x['optimizer'],
            'scheduler': x['scheduler']
        }

        return train_metrics, val_metrics