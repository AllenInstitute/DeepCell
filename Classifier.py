import logging
import os
import sys
import torch
from torch.utils.data import DataLoader

from DataSplitter import DataSplitter
from Metrics import Metrics, TrainingMetrics, CVMetrics
from RoiDataset import RoiDataset
from models.video.VideoNet import VideoNet

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, model: VideoNet, n_epochs: int, optimizer,
                 criterion, save_path, scheduler=None, scheduler_step_after_batch=False, debug=False,
                 early_stopping=30, use_cuda=True, seq_len=200):
        self.n_epochs = n_epochs
        self.model = model
        self.optimizer_constructor = optimizer
        self.scheduler_contructor = scheduler
        self.optimizer = optimizer()
        self.scheduler = scheduler(self.optimizer) if scheduler is not None else None
        self.scheduler_step_after_batch = scheduler_step_after_batch
        self.criterion = criterion
        self.use_cuda = use_cuda
        self.save_path = save_path
        self.debug = debug
        self.early_stopping = early_stopping
        self.seq_len = seq_len

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')

        if self.use_cuda:
            self.model.cuda()

        torch.save(self.model.state_dict(), f'{self.save_path}/model_init.pt')

    def cross_validate(self, train_dataset: RoiDataset, data_splitter: DataSplitter, batch_size=64, sampler=None,
                       n_splits=5, save_model=False, log_after_each_epoch=True):
        cv_metrics = CVMetrics(n_splits=n_splits)

        for i, (train, valid) in enumerate(data_splitter.get_cross_val_split(train_dataset=train_dataset,
                                                                             n_splits=n_splits)):
            logger.info(f'=========')
            logger.info(f'Fold {i}')
            logger.info(f'=========')

            train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size, sampler=sampler)
            valid_loader = DataLoader(dataset=valid, shuffle=False, batch_size=batch_size)
            train_metrics, valid_metrics = self.train(train_loader=train_loader, valid_loader=valid_loader,
                                                      save_model=save_model, eval_fold=i,
                                                      log_after_each_epoch=log_after_each_epoch)

            cv_metrics.update(train_metrics=train_metrics, valid_metrics=valid_metrics)

            self._reset()

        return cv_metrics

    def train(self, train_loader: DataLoader, eval_fold=None, valid_loader: DataLoader = None,
              log_after_each_epoch=True, save_model=False):
        all_train_metrics = TrainingMetrics(n_epochs=self.n_epochs)
        all_val_metrics = TrainingMetrics(n_epochs=self.n_epochs)

        best_epoch_metric = float('inf')
        time_since_best_epoch = 0

        for epoch in range(self.n_epochs):
            epoch_train_metrics = Metrics()
            epoch_val_metrics = Metrics()

            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data: torch.Tensor

                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                # reset RNN hidden state before each batch
                self.model.rnn.hidden_state = None

                # split movie into seq_len sized chunks (truncated backprop through time)
                chunks = data.split(split_size=self.seq_len, dim=2)

                logger.info(f'BATCH {batch_idx}')

                for i, c in enumerate(chunks):
                    logger.info(f'CHUNK {i}')
                    self.optimizer.zero_grad()
                    output = self.model(c)
                    output = output.squeeze()
                    loss = self.criterion(output, target.float())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                    self.optimizer.step()
                    self.model.rnn.detach_hidden_state()

                    epoch_train_metrics.update_loss(loss=loss.item(), num_batches=len(train_loader))
                    epoch_train_metrics.update_accuracies(y_true=target, y_out=output)

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
                        epoch_val_metrics.update_accuracies(y_true=target, y_out=output)

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       f1=epoch_val_metrics.F1)

                if epoch_val_metrics.loss < best_epoch_metric:
                    if save_model:
                        torch.save(self.model.state_dict(), f'{self.save_path}/{eval_fold}_model.pt')
                    all_train_metrics.best_epoch = epoch
                    all_val_metrics.best_epoch = epoch
                    best_epoch_metric = epoch_val_metrics.loss
                    time_since_best_epoch = 0
                else:
                    time_since_best_epoch += 1
                    if time_since_best_epoch > self.early_stopping:
                        logger.info('Stopping due to early stopping')
                        return all_train_metrics, all_val_metrics

            if not self.scheduler_step_after_batch:
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(epoch_val_metrics.loss)
                    else:
                        self.scheduler.step()

            if log_after_each_epoch:
                logger.info(f'Epoch: {epoch + 1} \tTrain Loss: {epoch_train_metrics.loss:.6f} '
                            f'\tVal loss: {epoch_val_metrics.loss:.6f}'
                            f'\tVal F1: {epoch_val_metrics.F1:.6f}')

        return all_train_metrics, all_val_metrics


    def _reset(self):
        # reset model weights
        state_dict = torch.load(f'{self.save_path}/model_init.pt')
        self.model.load_state_dict(state_dict)

        # reset optimizer
        self.optimizer = self.optimizer_constructor()

        # reset scheduler
        if self.scheduler is not None:
            self.scheduler = self.scheduler_contructor(self.optimizer)
