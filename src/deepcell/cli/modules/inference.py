import json
from pathlib import Path

import argschema
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from imgaug import augmenters as iaa

from deepcell.cli.schemas.inference import InferenceSchema
from deepcell.datasets.model_input import ModelInput
from deepcell.inference import inference
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.transform import Transform
from deepcell.models.classifier import Classifier


class InferenceModule(argschema.ArgSchemaParser):
    """
    Runs inference for experiment and produces csv of predictions
    """
    default_schema = InferenceSchema

    def run(self):
        test_transform = RoiDataset.get_default_transforms(
            crop_size=self.args['data_params']['crop_size'], is_train=False)

        model_inputs = self.args['model_inputs']
        test = RoiDataset(
            model_inputs=model_inputs,
            transform=test_transform
        )
        test_dataloader = DataLoader(dataset=test, shuffle=False,
                                     batch_size=self.args['batch_size'])

        model = getattr(
            torchvision.models,
            self.args['model_params']['model_architecture'])(
            pretrained=self.args['model_params']['use_pretrained_model'],
            progress=False)
        model = Classifier(
            model=model,
            truncate_to_layer=self.args['model_params']['truncate_to_layer'],
            classifier_cfg=self.args['model_params']['classifier_cfg'])
        _, inference_res = inference(
            model=model,
            test_loader=test_dataloader,
            has_labels=False,
            checkpoint_path=str(self.args['model_load_path']))

        if self.args['experiment_id'] is not None:
            out_filename = f'{self.args["experiment_id"]}_inference.csv'
        else:
            out_filename = f'inference.csv'
        inference_res.to_csv(Path(self.args['save_path']) / f'{out_filename}',
                             index=False)


def main():
    inference = InferenceModule()
    inference.run()


if __name__ == "__main__":
    main()
