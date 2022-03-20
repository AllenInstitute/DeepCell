from typing import Tuple, List

import argschema
import pandas as pd
import torch
import torchvision
from torchvision.transforms import transforms
import imgaug.augmenters as iaa

from deepcell.callbacks.early_stopping import EarlyStopping
from deepcell.cli.schemas.train import TrainSchema
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.models.classifier import Classifier
from deepcell.trainer import Trainer
from deepcell.transform import Transform


class TrainModule(argschema.ArgSchemaParser):
    default_schema = TrainSchema

    def run(self):
        dataset = self.args['data_params']['model_inputs']
        train_transform = RoiDataset.get_default_transforms(
            crop_size=self.args['data_params']['crop_size'], is_train=True)
        test_transform = RoiDataset.get_default_transforms(
            crop_size=self.args['data_params']['crop_size'], is_train=False)
        data_splitter = DataSplitter(model_inputs=dataset,
                                     train_transform=train_transform,
                                     test_transform=test_transform,
                                     seed=1234,
                                     image_dim=(128, 128),
                                     use_correlation_projection=True)
        train, test = data_splitter.get_train_test_split(
            test_size=self.args['test_fraction'])

        optimization_params = self.args['optimization_params']
        model_params = self.args['model_params']
        trainer = Trainer.from_model(
            model_architecture=model_params['model_architecture'],
            use_pretrained_model=model_params['use_pretrained_model'],
            classifier_cfg=model_params['classifier_cfg'],
            save_path=self.args['save_path'],
            n_epochs=optimization_params['n_epochs'],
            early_stopping_params=optimization_params['early_stopping_params'],
            dropout_prob=model_params['dropout_prob'],
            truncate_to_layer=model_params['truncate_to_layer'],
            freeze_to_layer=model_params['freeze_to_layer'],
            model_load_path=model_params['model_load_path'],
            learning_rate=optimization_params['learning_rate'],
            weight_decay=optimization_params['weight_decay'],
            learning_rate_scheduler=optimization_params['scheduler_params'],
        )
        trainer.cross_validate(
            train_dataset=train,
            data_splitter=data_splitter,
            batch_size=self.args['batch_size'],
            n_splits=self.args['n_folds']
        )
