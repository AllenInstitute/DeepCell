import time
from typing import Optional

import argschema

from deepcell.cli.schemas.train import TrainSchema
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.tracking.mlflow_utils import MLFlowTrackableMixin
from deepcell.trainer import Trainer


class TrainModule(argschema.ArgSchemaParser, MLFlowTrackableMixin):
    def __init__(self, input_data: Optional[dict] = None,
                 args: Optional[list] = None):
        self.default_schema = TrainSchema
        argschema.ArgSchemaParser().__init__(input_data=input_data, args=args)
        MLFlowTrackableMixin().__init__(
            server_uri=self.args['tracking_params']['mlflow_server_uri'],
            experiment_name=
            self.args['tracking_params']['mlflow_experiment_name']
        )

    def run(self):
        if self._is_mlflow_tracking_enabled:
            self._create_parent_mlflow_run(run_name=f'CV-{int(time.time())}')

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
        tracking_params = self.args['tracking_params']
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
            model_load_path=self.args['model_load_path'],
            learning_rate=optimization_params['learning_rate'],
            weight_decay=optimization_params['weight_decay'],
            learning_rate_scheduler=optimization_params['scheduler_params'],
            mlflow_server_uri=tracking_params['mlflow_server_uri'],
            mlflow_experiment_name=tracking_params['mlflow_experiment_name']
        )
        trainer.cross_validate(
            train_dataset=train,
            data_splitter=data_splitter,
            batch_size=self.args['batch_size'],
            n_splits=self.args['n_folds']
        )
