import logging
import random
import sys

import numpy as np
import torch

from deepcell.data_splitter import DataSplitter
from deepcell.classifier import Classifier
from roi_dataset import RoiDataset

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
CLASSIFIER_CONFIG = [512]
CRITERION = torch.nn.BCEWithLogitsLoss


class HyperparamTuner:
    def __init__(self, model: torch.nn.Module, param_distributions: ParamDistribution, train_dataset: RoiDataset,
                 data_splitter: DataSplitter, batch_size=64, sampler=None, iters=10, n_cv_splits=5,
                 early_stopping=10):
        self.model = model
        self.param_distributions = param_distributions
        self.train_dataset = train_dataset
        self.data_splitter = data_splitter
        self.batch_size = batch_size
        self.sampler = sampler
        self.iters = iters
        self.n_cv_splits = n_cv_splits
        self.early_stopping = early_stopping

    def search(self, n_epochs=1000):
        res = []
        for iter in range(self.iters):
            params = self.param_distributions.sample()

            model_params = params['model'] if 'model' in params else {}
            optimizer_params = params['optimizer'] if 'optimizer' in params else {}
            scheduler_params = params['scheduler'] if 'scheduler' in params else {}
            criterion_params = params['criterion'] if 'criterion' in params else {}

            model = self.model(**model_params['params'])

            if optimizer_params:
                optimizer = lambda: optimizer_params['optimizer'](optimizer_params['optimizer_params'](model),
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
                save_path='./saved_models',
                early_stopping=self.early_stopping
            )

            logger.info('==========')
            logger.info(f'ITER {iter}')
            logger.info('==========')
            logger.info(params)
            logger.info('Cross validating')

            cv_metrics = classifier.cross_validate(train_dataset=self.train_dataset,
                                                   data_splitter=self.data_splitter,
                                                   batch_size=self.batch_size,
                                                   sampler=self.sampler, n_splits=self.n_cv_splits,
                                                   save_model=False)
            d = {}
            for category, category_params in params.items():
                params = category_params['params']
                for param, val in params.items():
                    d[param] = val

            metrics = cv_metrics.metrics
            for k, v in metrics.items():
                d[k] = v
            res.append(d)
        return res
