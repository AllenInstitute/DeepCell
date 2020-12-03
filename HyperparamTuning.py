import logging
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.CNN import CNN
from Classifier import Classifier
from Plotting import Plotting
from SlcDataset import SlcDataset

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(levelname)s:\t%(asctime)s\t%(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p'
)

logger = logging.getLogger(__name__)


class ParamDistribution:
    def __init__(self, param_distribution):
        """

        :param param_distribution:
            eg ['model': {name: ..., distr: ..., distr_type: ...}, {...}]
        """
        self.param_distribution = param_distribution

    def sample(self):
        res = {}
        for category in self.param_distribution:
            category_params = self.param_distribution[category]
            params = category_params['params']
            res[category] = {}
            res[category]['params'] = {param: self._sample_param(param=params[param]) for param in params}
            others = set(category_params.keys()).difference(['params'])
            for other in others:
                res[category][other] = category_params[other]
        return res

    def _sample_param(self, param):
        distr = param['distr']

        if param['distr_type'] == 'DISCRETE':
            return random.choice(distr)
        elif param['distr_type'] == 'LOG_SCALE':
            low, high = distr
            return 10 ** np.random.uniform(low=low, high=high)
        elif param['distr_type'] == 'LINEAR_SCALE':
            low, high = distr
            return np.random.uniform(low=low, high=high)
        else:
            return distr


OPTIMIZER = torch.optim.Adam
LR = 1e-3
CONV_CONFIG = [32, 32, 'M', 64, 64, 'M']
CLASSIFIER_CONFIG = [512]
CRITERION = torch.nn.BCEWithLogitsLoss


class HyperparamTuner:
    def __init__(self, param_distributions: ParamDistribution, train_loader: DataLoader, valid_loader: DataLoader,
                 iters=10):
        self.param_distributions = param_distributions
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.iters = iters

    def search(self, train_dataset: SlcDataset, n_epochs=100):
        res = []
        for iter in range(self.iters):
            params = self.param_distributions.sample()

            model_params = params['model'] if 'model' in params else \
                {'params': {'conv_cfg': CONV_CONFIG, 'classifier_cfg': CLASSIFIER_CONFIG}}
            optimizer_params = params['optimizer'] if 'optimizer' in params else {}
            scheduler_params = params['scheduler'] if 'scheduler' in params else {}
            criterion_params = params['criterion'] if 'criterion' in params else {}

            if 'conv_cfg' not in model_params['params']:
                model_params['params']['conv_cfg'] = CONV_CONFIG
            if 'classifier_cfg' not in model_params['params']:
                model_params['classifier_cfg'] = CLASSIFIER_CONFIG
            model = CNN(**model_params['params'])

            if optimizer_params:
                optimizer = lambda: optimizer_params['optimizer'](model.parameters(),
                                                                  **optimizer_params['params'])
            else:
                optimizer = lambda: OPTIMIZER(model.parameters(), lr=LR)

            if scheduler_params:
                scheduler = lambda optimizer: scheduler_params['scheduler'](optimizer, **scheduler_params['params'])
            else:
                scheduler = None

            criterion = CRITERION(**criterion_params['params']) if criterion_params else CRITERION()
            classifier = Classifier(
                model=model,
                n_epochs=n_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                save_path='./saved_models'
            )

            logger.info('==========')
            logger.info(f'ITER {iter}')
            logger.info('==========')
            logger.info(params)
            logger.info('Cross validating')

            train_metrics, val_metrics = classifier.train(train_loader=self.train_loader,
                                                          valid_loader=self.valid_loader)
            d = {}
            for category, category_params in params.items():
                params = category_params['params']
                for param, val in params.items():
                    d[param] = val
            d['precision'] = val_metrics.precisions[-1]
            d['recall'] = val_metrics.recalls[-1]
            d['f1'] = val_metrics.f1s[-1]

            plotting = Plotting(experiment_name=iter)
            fig = plotting.plot_loss(train_loss=train_metrics.losses, val_loss=val_metrics.losses)
            d['loss_plot'] = f'results/loss_{iter}_{plotting.file_suffix}.png'
            fig.write_image(d['loss_plot'])

            fig = plotting.plot_train_val_F1(train_f1=train_metrics.f1s, val_f1=val_metrics.f1s)
            d['f1_plot'] = f'results/f1_{iter}_{plotting.file_suffix}.png'
            fig.write_image(d['f1_plot'])

            res.append(d)
        return res
