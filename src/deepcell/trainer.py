import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, Callable, List

import torch
from torch.utils.data import DataLoader

from deepcell.data_splitter import DataSplitter
from deepcell.metrics import ClassificationMetrics, \
    CVMetrics, TrainingMetrics, LocalizationMetrics, Metrics
from deepcell.datasets.roi_dataset import RoiDataset

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

logger = logging.getLogger(__name__)


class Trainer:
    """Handles train loop"""
    def __init__(
            self,
            model: torch.nn.Module,
            n_epochs: int,
            optimizer: Callable[..., torch.optim.Adam],
            criterion: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
            save_path: Union[str, Path],
            scheduler: Optional[
                Callable[[torch.optim.Adam],
                         torch.optim.lr_scheduler.ReduceLROnPlateau]] = None,
            scheduler_step_after_batch=False,
            early_stopping=30,
            early_stopping_larger_is_better=False,
            model_load_path: Optional[Union[str, Path]] = None,
            task='classification',
            additional_metrics: List = None,
            early_stopping_metric: Optional[str] = None,
    ):
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
            task
                classification or localization
            additional_metrics
                In addition to the base metrics, what other metrics need to
                be tracked
            early_stopping_metric
                The metric to use for early stopping
                If classification, defaults to F1
                If localization, defaults to loss
            early_stopping_larger_is_better
                Whether early_stopping_metric increasing is a good thing
        """
        if task not in ('classification', 'localization'):
            raise ValueError('Invalid task. Valid tasks are classification '
                             'or localization')
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
        self._early_stopping_metric_larger_is_better = \
            early_stopping_larger_is_better
        self.model_load_path = model_load_path
        self.save_path = save_path
        self._current_best_state_dicts = {}
        self._task = task
        self._additional_metrics = \
            additional_metrics if additional_metrics is not None else {}

        if early_stopping_metric is None:
            if task == 'classification':
                early_stopping_metric = 'F1'
            else:
                early_stopping_metric = 'loss'
        self._early_stopping_metric = early_stopping_metric

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
                train_metrics = None
                val_metrics = None

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
            all_train_metrics = TrainingMetrics(
                n_epochs=self.n_epochs,
                best_metric=self._early_stopping_metric,
                metric_larger_is_better=
                self._early_stopping_metric_larger_is_better,
                additional_metrics={m: None for m in self._additional_metrics}
            )

        if val_metrics is not None:
            all_val_metrics = val_metrics
        else:
            all_val_metrics = TrainingMetrics(
                n_epochs=self.n_epochs,
                best_metric=self._early_stopping_metric,
                metric_larger_is_better=
                self._early_stopping_metric_larger_is_better,
                additional_metrics={m: None for m in self._additional_metrics}
            )

        time_since_best_epoch = 0

        for epoch in range(all_val_metrics.best_epoch + 1, self.n_epochs):
            epoch_train_metrics = \
                ClassificationMetrics() if self._task == 'classification' \
                else LocalizationMetrics()
            epoch_val_metrics = \
                ClassificationMetrics() if self._task == 'classification' \
                else LocalizationMetrics()

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

                self._update_metrics(epoch_metrics=epoch_train_metrics,
                                     history=all_train_metrics, epoch=epoch)
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

                self._update_metrics(epoch_metrics=epoch_val_metrics,
                                     history=all_val_metrics, epoch=epoch)

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

            logger_str = [f'Epoch: {epoch + 1}',
                          f'Train Loss: {epoch_train_metrics.loss:.3f}',
                          f'Val Loss: {epoch_val_metrics.loss:.3f}']

            for m in self._additional_metrics:
                train_val = getattr(epoch_train_metrics, m, None)
                val_val = getattr(epoch_val_metrics, m, None)
                if train_val is None or val_val is None:
                    raise ValueError(f'Cannot log {m} because it is not '
                                     f'tracked')
                logger_str.append(f'Train {m}: {train_val:.3f}')
                logger_str.append(f'Val {m}: {val_val:.3f}')

            if log_after_each_epoch:
                logger.info('\t'.join(logger_str))
        self._save_model_and_performance(eval_fold=eval_fold,
                                         all_train_metrics=all_train_metrics,
                                         all_val_metrics=all_val_metrics,
                                         epoch=self.n_epochs)

        return all_train_metrics, all_val_metrics

    def _update_metrics(self, epoch_metrics: Metrics,
                        history: TrainingMetrics, epoch: int):
        early_stopping_metric_val = getattr(epoch_metrics,
                                            history.best_metric)

        additional_metrics = {
            metric: getattr(epoch_metrics, metric) for metric in
            self._additional_metrics}
        history.update(epoch=epoch,
                       loss=epoch_metrics.loss,
                       best_metric_val=early_stopping_metric_val,
                       additional_metrics=additional_metrics)

    def _reset(self):
        # reset model weights
        x = torch.load(f'{self.save_path}/model_init.pt', map_location='cpu')
        if torch.cuda.is_available():
            x = x.cuda()
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
            additional_metrics={metric: x['performance']['train'][metric]
                                for metric in self._additional_metrics},
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
            additional_metrics={metric: x['performance']['val'][metric]
                                for metric in self._additional_metrics},
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
