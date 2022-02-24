from typing import Tuple

import argschema
import torch
import torchvision
from torchvision.transforms import transforms
import imgaug.augmenters as iaa

from deepcell.callbacks.early_stopping import EarlyStopping
from deepcell.cli.schemas.train import TrainSchema
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.visual_behavior_extended_dataset import \
    VisualBehaviorExtendedDataset
from deepcell.models.classifier import Classifier
from deepcell.trainer import Trainer
from deepcell.transform import Transform


class TrainModule(argschema.ArgSchemaParser):
    default_schema = TrainSchema

    def run(self):
        dataset = VisualBehaviorExtendedDataset(
            artifact_destination=self.args['data_params']['download_path'],
            exclude_projects=self.args['data_params']['exclude_projects'])
        train_transform, test_transform = self._get_transforms()
        data_splitter = DataSplitter(model_inputs=dataset.dataset,
                                     train_transform=train_transform,
                                     test_transform=test_transform, seed=1234,
                                     image_dim=(128, 128),
                                     use_correlation_projection=True)
        train, test = data_splitter.get_train_test_split(
            test_size=self.args['test_fraction'])

        model = getattr(
            torchvision.models,
            self.args['model_params']['model_architecture'])(
            pretrained=self.args['model_params']['use_pretrained_model'],
            progress=False)

        model = Classifier(
            model=model,
            truncate_to_layer=self.args['model_params']['truncate_to_layer'],
            freeze_up_to_layer=self.args['model_params']['freeze_to_layer'],
            classifier_cfg=self.args['model_params']['classifier_cfg'],
            dropout_prob=self.args['model_params']['dropout_prob'])

        trainer = self._get_trainer(model=model)
        trainer.cross_validate(
            train_dataset=train,
            data_splitter=data_splitter,
            batch_size=self.args['batch_size'],
            n_splits=self.args['n_folds']
        )

    def _get_trainer(self, model: torch.nn.Module) -> Trainer:
        optimization_params = self.args['optimization_params']
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimization_params['learning_rate'],
            weight_decay=optimization_params['weight_decay'])
        if optimization_params['scheduler_params']['type'] is not None:
            scheduler = getattr(
                torch.optim.lr_scheduler,
                optimization_params['scheduler_params']['type'])(
                optimizer=optimizer,
                mode='min',
                patience=optimization_params['scheduler_params']['patience'],
                factor=optimization_params['scheduler_params']['factor'],
                verbose=True
            )
        else:
            scheduler = None
        trainer = Trainer(
            model=model,
            n_epochs=optimization_params['n_epochs'],
            criterion=torch.nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            save_path=self.args['save_path'],
            scheduler=scheduler,
            callbacks=[
                EarlyStopping(
                    best_metric=optimization_params['early_stopping_params'][
                        'monitor'],
                    patience=optimization_params['early_stopping_params'][
                        'patience']
                )
            ],
            model_load_path=self.args['model_load_path']
        )
        return trainer

    def _get_transforms(self) -> Tuple[Transform, Transform]:
        width, height = self.args['data_params']['crop_size']
        all_transform = transforms.Compose([
            iaa.Sequential([
                iaa.Affine(
                    rotate=[0, 90, 180, 270, -90, -180, -270], order=0
                ),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.CenterCropToFixedSize(height=height, width=width),
            ]).augment_image,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        train_transform = Transform(all_transform=all_transform)

        all_transform = transforms.Compose([
            iaa.Sequential([
                iaa.CenterCropToFixedSize(height=height, width=width)
            ]).augment_image,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_transform = Transform(all_transform=all_transform)
        return train_transform, test_transform
