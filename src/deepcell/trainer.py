import os
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from deepcell.callbacks.base_callback import Callback
from deepcell.callbacks.early_stopping import EarlyStopping
from deepcell.data_splitter import DataSplitter
from deepcell.metrics import Metrics, CVMetrics
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.models.classifier import Classifier
from deepcell.logger import init_logger


logger = init_logger(__name__)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            n_epochs: int,
            criterion: torch.nn.BCEWithLogitsLoss,
            optimizer: torch.optim.Adam,
            save_path: Union[str, Path],
            scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None, # noqa E501
            callbacks: Optional[List[Callback]] = None,
            model_load_path: Optional[Union[str, Path]] = None):
        """
        The driver for the training and evaluation loop
        Args:
            model:
                torch.nn.Module
            n_epochs:
                Number of epochs to train for
            optimizer
                Optimizer instance.
                See torch.optim for options
            criterion:
                loss function
            save_path:
                Where to save checkpoints
            scheduler
                Optional learning rate decay scheduler
                See torch.optim.lr_scheduler package for options
            model_load_path:
                Path to load a pretrained model. Activates continuation of
                training
        """
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()
        self.model_load_path = model_load_path
        self.save_path = save_path
        self._current_best_state_dicts = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if scheduler is not None else None # noqa E501
        }
        self._callback_metrics = {}
        self._callbacks = callbacks if callbacks is not None else []

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')

        if self.use_cuda:
            self.model = self.model.cuda()

        self._save_model_and_performance(is_init=True)

    @property
    def early_stopping_callback(self) -> Optional[EarlyStopping]:
        for c in self._callbacks:
            if isinstance(c, EarlyStopping):
                return c
        return None

    def cross_validate(self, train_dataset: RoiDataset, data_splitter: DataSplitter, batch_size=64, sampler=None,
                       n_splits=5, log_after_each_epoch=True):
        cv_metrics = CVMetrics(n_splits=n_splits)

        for i, (train, valid) in enumerate(data_splitter.get_cross_val_split(train_dataset=train_dataset,
                                                                             n_splits=n_splits)):
            logger.info(f'=========')
            logger.info(f'Fold {i}')
            logger.info(f'=========')

            train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size, sampler=sampler)
            valid_loader = DataLoader(dataset=valid, shuffle=False, batch_size=batch_size)
            self.train(
                train_loader=train_loader, valid_loader=valid_loader,
                eval_fold=i,
                log_after_each_epoch=log_after_each_epoch)

            best_epoch = self.early_stopping_callback.best_epoch if \
                self.early_stopping_callback else None
            cv_metrics.update(metrics=self._callback_metrics,
                              best_epoch=best_epoch)

            self._reset()

        return cv_metrics

    def train(self, train_loader: DataLoader, eval_fold=None, valid_loader: DataLoader = None,
              log_after_each_epoch=True):
        if self.model_load_path is not None:
            self._load_pretrained_model(
                checkpoint_path=
                Path(self.model_load_path) / f'{eval_fold}_model.pt' if
                eval_fold is not None else
                Path(self.model_load_path) / 'model.pt')

        if not self._callback_metrics:
            self._callback_metrics = {
                'loss': np.zeros(self.n_epochs),
                'f1': np.zeros(self.n_epochs),
                'val_loss': np.zeros(self.n_epochs),
                'val_f1': np.zeros(self.n_epochs)
            }

        start_epoch = self.early_stopping_callback.best_epoch + 1 if \
            self.early_stopping_callback else 0
        for epoch in range(start_epoch, self.n_epochs):
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

            self._callback_metrics['loss'][epoch] = epoch_train_metrics.loss
            self._callback_metrics['f1'][epoch] = epoch_train_metrics.F1

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

                self._callback_metrics['val_loss'][epoch] = \
                    epoch_val_metrics.loss
                self._callback_metrics['val_f1'][epoch] = \
                    epoch_val_metrics.F1

                if self.early_stopping_callback is not None:
                    self.early_stopping_callback.on_epoch_end(
                        epoch=epoch, metrics=self._callback_metrics)
                    if self.early_stopping_callback.best_epoch == epoch:
                        scheduler_state_dict = self.scheduler.state_dict() if \
                            self.scheduler is not None else None
                        self._current_best_state_dicts = {
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': scheduler_state_dict
                        }
                        self.early_stopping_callback.time_since_best_epoch = 0
                    else:
                        self.early_stopping_callback.time_since_best_epoch += 1
                        if self.early_stopping_callback.time_since_best_epoch \
                                > self.early_stopping_callback.patience:
                            logger.info('Stopping due to early stopping')
                            self._save_model_and_performance(
                                eval_fold=eval_fold)
                            return

            if self.scheduler is not None:
                self.scheduler.step(epoch_val_metrics.loss)

            if log_after_each_epoch:
                logger.info(f'Epoch: {epoch + 1} \t'
                            f'Train F1: {epoch_train_metrics.F1:.6f} \t'
                            f'Val F1: {epoch_val_metrics.F1:.6f}\t'
                            f'Train Loss: {epoch_train_metrics.loss:.6f}\t'
                            f'Val Loss: {epoch_val_metrics.loss}'
                            )
        self._save_model_and_performance(eval_fold=eval_fold)

    @staticmethod
    def from_model(
            model_architecture: str,
            use_pretrained_model: bool,
            classifier_cfg: List[int],
            save_path: Path,
            n_epochs: int,
            learning_rate: float,
            early_stopping_params: Dict,
            dropout_prob: float = 0.0,
            truncate_to_layer: Optional[int] = None,
            freeze_to_layer: Optional[int] = None,
            model_load_path: Optional[Path] = None,
            weight_decay: float = 0.0,
            learning_rate_scheduler: Optional[Dict] = None,
    ) -> "Trainer":
        model = getattr(
            torchvision.models,
            model_architecture)(
            pretrained=use_pretrained_model,
            progress=False)

        model = Classifier(
            model=model,
            truncate_to_layer=truncate_to_layer,
            freeze_up_to_layer=freeze_to_layer,
            classifier_cfg=classifier_cfg,
            dropout_prob=dropout_prob)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay)
        if learning_rate_scheduler is not None:
            scheduler = getattr(
                torch.optim.lr_scheduler,
                learning_rate_scheduler['type'])(
                optimizer=optimizer,
                mode='min',
                patience=learning_rate_scheduler['patience'],
                factor=learning_rate_scheduler['factor'],
                verbose=True
            )
        else:
            scheduler = None
        trainer = Trainer(
            model=model,
            n_epochs=n_epochs,
            criterion=torch.nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            save_path=save_path,
            scheduler=scheduler,
            callbacks=[
                EarlyStopping(
                    best_metric=early_stopping_params['monitor'],
                    patience=early_stopping_params['patience']
                )
            ],
            model_load_path=model_load_path
        )
        return trainer

    def _reset(self):
        # reset model weights
        x = torch.load(f'{self.save_path}/init_model.pt')
        self.model.load_state_dict(x['state_dict'])

        # reset optimizer
        self.optimizer.load_state_dict(x['optimizer'])

        # reset scheduler
        if self.scheduler is not None:
            self.scheduler.load_state_dict(x['scheduler'])

    def _save_model_and_performance(self, eval_fold: Optional[int] = None,
                                    is_init=False):
        """Writes model weights and training performance to disk

        Args:
            eval_fold:
                The validation fold to save a checkpoint for
            is_init:
                Whether this is before training has started. Used to save a
                checkpoint to reset to after training on a fold has finished
        """
        metrics = {
            k: v[:self.early_stopping_callback.best_epoch + 1] if
            self.early_stopping_callback else v for k, v in
            self._callback_metrics.items()}
        if is_init:
            checkpoint_name = 'init_model'
        elif eval_fold is not None:
            checkpoint_name = f'{eval_fold}_model'
        else:
            checkpoint_name = 'model'
        d = {
            'state_dict': self._current_best_state_dicts['model'],
            'optimizer': self._current_best_state_dicts['optimizer'],
            'scheduler': self._current_best_state_dicts['scheduler'],
            'performance': metrics,

        }
        if self.early_stopping_callback is not None:
            d['early_stopping'] = self.early_stopping_callback.to_dict()
        torch.save(d, f'{self.save_path}/{checkpoint_name}.pt')

    def _load_pretrained_model(self, checkpoint_path: Path):
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

        self._callback_metrics = {
            k: v.tolist() + [0] * self.n_epochs for k, v in
            x['performance'].items()
        }

        self._current_best_state_dicts = {
            'model': x['state_dict'],
            'optimizer': x['optimizer'],
            'scheduler': x['scheduler']
        }

        if self.early_stopping_callback is not None and 'early_stopping' in x:
            if self.early_stopping_callback.best_metric != \
                    x['early_stopping']['best_metric']:
                raise ValueError(
                    f'Trying to use '
                    f'{self.early_stopping_callback.best_metric} '
                    f'for early stopping but '
                    f'{x["early_stopping"]["best_metric"]} was used '
                    f'previously')
            self.early_stopping_callback.best_epoch = \
                x['early_stopping']['best_epoch']
            self.early_stopping_callback.best_metric_value = \
                x['early_stopping']['best_metric_value']