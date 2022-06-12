from typing import Optional

import argschema
from torch.utils.data import DataLoader

from deepcell.cli.schemas.train import TrainSchema
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.logger import init_logger
from deepcell.trainer import Trainer


class TrainRunner(argschema.ArgSchemaParser):
    def __init__(self, input_data: Optional[dict] = None,
                 args: Optional[list] = None):
        super().__init__(input_data=input_data, args=args,
                         schema_type=TrainSchema)
        self._logger = \
            init_logger(name=__name__, log_path=self.args['log_path'])

    def run(self):
        self._logger.info(
            {k: v for k, v in self.args.items()
             # Don't print this (too big)
             if not k.endswith('model_inputs')})

        train = self.args['train_model_inputs']
        validation = self.args['validation_model_inputs']

        train_transform = RoiDataset.get_default_transforms(
            crop_size=self.args['data_params']['crop_size'], is_train=True,
            means=self.args['data_params']['channel_wise_means'],
            stds=self.args['data_params']['channel_wise_stds']
        )
        test_transform = RoiDataset.get_default_transforms(
            crop_size=self.args['data_params']['crop_size'], is_train=False,
            means=self.args['data_params']['channel_wise_means'],
            stds=self.args['data_params']['channel_wise_stds'],
        )

        train = RoiDataset(
            model_inputs=train,
            transform=train_transform
        )
        validation = RoiDataset(
            model_inputs=validation,
            transform=test_transform
        )

        train_loader = DataLoader(dataset=train, shuffle=True,
                                  batch_size=self.args['batch_size'])
        valid_loader = DataLoader(dataset=validation, shuffle=False,
                                  batch_size=self.args['batch_size'])

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
        trainer.train(
            train_loader=train_loader, valid_loader=valid_loader,
            eval_fold=self.args['fold'],
            sagemaker_job_name=tracking_params['sagemaker_job_name'],
            mlflow_parent_run_id=tracking_params['parent_run_id']
        )


if __name__ == '__main__':
    train_runner = TrainRunner()
    train_runner.run()
