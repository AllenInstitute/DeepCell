import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
from torch.utils.data import DataLoader

from DataSplitter import DataSplitter
from Metrics import Metrics, TrainingMetrics, CVMetrics
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
                 criterion, save_path, scheduler=None, scheduler_step_after_batch=False, debug=False,
                 early_stopping=30):
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
        self.early_stopping = early_stopping

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')

        if self.use_cuda:
            self.model.cuda()

        torch.save(self.model.state_dict(), f'{self.save_path}/model_init.pt')

    def cross_validate(self, train_dataset: SlcDataset, data_splitter: DataSplitter, batch_size=64, sampler=None,
                       n_splits=5, save_model=False):
        cv_metrics = CVMetrics(n_splits=n_splits, n_epochs=self.n_epochs)

        for i, (train, valid) in enumerate(data_splitter.get_cross_val_split(train_dataset=train_dataset,
                                                                             n_splits=n_splits)):
            logger.info(f'=========')
            logger.info(f'Fold {i}')
            logger.info(f'=========')

            train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size, sampler=sampler)
            valid_loader = DataLoader(dataset=valid, shuffle=False, batch_size=batch_size)
            train_metrics, valid_metrics = self.train(train_loader=train_loader, valid_loader=valid_loader,
                                                      save_model=save_model, eval_fold=i)

            cv_metrics.update(train_metrics=train_metrics, valid_metrics=valid_metrics)

            self._reset()

        return cv_metrics

    def train(self, train_loader: DataLoader, eval_fold=None, valid_loader: DataLoader = None,
              log_after_each_epoch=True, save_model=False):
        all_train_metrics = TrainingMetrics(n_epochs=self.n_epochs)
        all_val_metrics = TrainingMetrics(n_epochs=self.n_epochs)

        best_epoch_f1 = -float('inf')
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

                if epoch_val_metrics.F1 > best_epoch_f1:
                    if save_model:
                        torch.save(self.model.state_dict(), f'{self.save_path}/{eval_fold}_model.pt')
                    best_epoch = epoch
                    best_epoch_f1 = epoch_val_metrics.F1
                    time_since_best_epoch = 0
                else:
                    time_since_best_epoch += 1
                    if time_since_best_epoch > self.early_stopping:
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

    def test(self, test_loader: DataLoader, has_labels=True):
        dataset: SlcDataset = test_loader.dataset
        metrics = Metrics()

        models = os.listdir(f'./saved_models/checkpoints')
        models = [model for model in models if model != 'model_init.pt']

        y_scores = np.zeros((len(models), len(dataset)))
        y_preds = np.zeros((len(models), len(dataset)))

        for i, model_checkpoint in enumerate(models):
            state_dict = torch.load(f'./saved_models/checkpoints/{model_checkpoint}')
            self.model.load_state_dict(state_dict)

            self.model.eval()
            prev_start = 0

            for data, _ in test_loader:
                data = data.cuda()

                with torch.no_grad():
                    output = self.model(data)
                    output = output.squeeze()
                    y_score = torch.sigmoid(output).cpu().numpy()
                    start = prev_start
                    end = start + data.shape[0]
                    y_scores[i][start:end] = y_score
                    y_preds[i][start:end] = y_score > .5
                    prev_start = end

            if has_labels:
                TP = ((dataset.y == 1) & (y_preds[i] == 1)).sum().item()
                FP = ((dataset.y == 0) & (y_preds[i] == 1)).sum().item()
                FN = ((dataset.y == 1) & (y_preds[i] == 0)).sum().item()
                p = TP / (TP + FP)
                r = TP / (TP + FN)
                f1 = 2 * p * r / (p + r)
                print(f'{model_checkpoint} precision: {p}')
                print(f'{model_checkpoint} recall: {r}')
                print(f'{model_checkpoint} f1: {f1}')

        y_scores = y_scores.mean(axis=0)
        y_preds = y_scores > .5
        if has_labels:
            metrics.update_accuracies(y_true=dataset.y, y_score=y_scores)

        df = pd.DataFrame({'roi-id': dataset.roi_ids, 'y_score': y_scores, 'y_pred': y_preds})

        return metrics, df

    def _reset(self):
        # reset model weights
        state_dict = torch.load(f'{self.save_path}/model_init.pt')
        self.model.load_state_dict(state_dict)

        # reset optimizer
        self.optimizer = self.optimizer_constructor()

        # reset scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler_contructor(self.optimizer)
