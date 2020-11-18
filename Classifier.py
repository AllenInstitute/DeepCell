import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from KfoldDataLoader import KfoldDataLoader
from Metrics import Metrics
from TrainingMetrics import TrainingMetrics

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, model: torch.nn.Module, n_epochs: int, test_loader: DataLoader, optimizer, scheduler, criterion,
                 save_path, train_loader: DataLoader = None, kfoldDataLoader: KfoldDataLoader = None, debug=False,
                 use_learning_rate_scheduler=False):
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer_constructor = optimizer
        self.scheduler_contructor = scheduler
        self.optimizer = optimizer()
        self.scheduler = scheduler(self.optimizer)
        self.criterion = criterion
        self.use_cuda = torch.cuda.is_available()
        self.save_path = save_path
        self.kfoldDataLoader = kfoldDataLoader
        self.debug = debug
        self.use_learning_rate_scheduler = use_learning_rate_scheduler

        if not os.path.exists(f'{self.save_path}'):
            os.makedirs(f'{self.save_path}')

        if self.use_cuda:
            self.model.cuda()

        torch.save(self.model.state_dict(), f'{self.save_path}/model_init.pt')

    def cross_validate(self):
        if self.debug:
            data_loaders = [(self.train_loader, None)]
            n_folds = 1
        else:
            data_loaders = self.kfoldDataLoader.run()
            n_folds = self.kfoldDataLoader.n_splits

        train_losses = np.zeros((n_folds, self.n_epochs))
        val_losses = np.zeros((n_folds, self.n_epochs))
        precisions = np.zeros((n_folds, self.n_epochs))
        recalls = np.zeros((n_folds, self.n_epochs))
        train_f1s = np.zeros((n_folds, self.n_epochs))
        val_f1s = np.zeros((n_folds, self.n_epochs))
        best_epochs = np.zeros(n_folds, dtype=np.int)

        logger.info(f'Train evaluate on {n_folds} folds')

        for i, (train_loader, valid_loader) in enumerate(data_loaders):
            logger.info(f'Train/evaluate on fold {i}')

            train_metrics, val_metrics = self._train(
                n_epochs=self.n_epochs, train_loader=train_loader, valid_loader=valid_loader,
                save_model=False
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


        best_epoch_precisions = np.array(
            [precisions[row][col] for row, col in zip(range(precisions.shape[0]), best_epochs)]
        )
        best_epoch_recalls = np.array(
            [recalls[row][col] for row, col in zip(range(recalls.shape[0]), best_epochs)]
        )
        best_epoch_f1s = np.array(
            [val_f1s[row][col] for row, col in zip(range(val_f1s.shape[0]), best_epochs)]
        )

        res = {
            'all_val_precision': precisions.mean(axis=0),
            'all_val_recall': recalls.mean(axis=0),
            'all_val_f1': val_f1s.mean(axis=0),
            'all_train_f1': train_f1s.mean(axis=0),
            'train_losses': train_losses.mean(axis=0),
            'val_losses': val_losses.mean(axis=0),
            'mean_val_best_epoch_precision': best_epoch_precisions.mean(),
            'std_val_best_epoch_precision': best_epoch_precisions.std(),
            'mean_val_best_epoch_recall': best_epoch_recalls.mean(),
            'std_val_best_epoch_recall': best_epoch_recalls.std(),
            'mean_val_best_epoch_f1': best_epoch_f1s.mean(),
            'std_val_best_epoch_f1': best_epoch_f1s.std(),
        }

        logger.info(f'Done train/evaluate for {n_folds} folds')

        logger.info({
            'mean_val_best_epoch_precision': res['mean_val_best_epoch_precision'],
            'std_val_best_epoch_precision': res['std_val_best_epoch_precision'],
            'mean_val_best_epoch_recall': res['mean_val_best_epoch_recall'],
            'std_val_best_epoch_recall': res['std_val_best_epoch_recall'],
            'mean_val_best_epoch_f1': res['mean_val_best_epoch_f1'],
            'std_val_best_epoch_f1': res['std_val_best_epoch_f1']
        })

        return res

    def _train(self, n_epochs, train_loader: DataLoader, valid_loader: DataLoader = None, save_model=False):
        all_train_metrics = TrainingMetrics(n_epochs=n_epochs)
        all_val_metrics = TrainingMetrics(n_epochs=n_epochs)

        for epoch in range(n_epochs):
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

                        if self.use_learning_rate_scheduler:
                            self.scheduler.step(loss)

                        epoch_val_metrics.update_loss(loss=loss.item(), num_batches=len(valid_loader))
                        epoch_val_metrics.update_accuracies(y_true=target, y_out=output)

                all_val_metrics.update(epoch=epoch,
                                       loss=epoch_val_metrics.loss,
                                       precision=epoch_val_metrics.precision,
                                       recall=epoch_val_metrics.recall,
                                       f1=epoch_val_metrics.F1)

            logger.info(f'Epoch: {epoch + 1} \tTrain Loss: {epoch_train_metrics.loss:.6f} '
                        f'\tTrain F1: {epoch_train_metrics.F1:.6f}'
                        f'\tVal F1: {epoch_val_metrics.F1:.6f}')

        if save_model:
            torch.save(self.model.state_dict(), f'{self.save_path}/model.pt')

        return all_train_metrics, all_val_metrics

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
                y_score = torch.sigmoid(output)
                y_pred = y_score > .5
                TP += ((target == 1) & (y_pred == 1)).sum().item()
                FN += ((target == 1) & (y_pred == 0)).sum().item()
                FP += ((target == 0) & (y_pred == 1)).sum().item()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')

    def _reset(self):
        # reset model weights
        state_dict = torch.load(f'{self.save_path}/model_init.pt')
        self.model.load_state_dict(state_dict)

        # reset optimizer
        self.optimizer = self.optimizer_constructor()

        # reset scheduler
        self.scheduler = self.scheduler_contructor(self.optimizer)

