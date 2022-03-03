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
        all_transform = transforms.Compose([
            iaa.Sequential([
                iaa.CenterCropToFixedSize(
                    width=self.args['data_params']['crop_size'][0],
                    height=self.args['data_params']['crop_size'][1]),
            ]).augment_image,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = Transform(all_transform=all_transform)

        model_inputs = self.args['data_params']['model_inputs']
        test = RoiDataset(
            model_inputs=model_inputs,
            transform=test_transform,
            use_correlation_projection=True,
            center_roi_centroid=self.args['data_params']['center_roi_centroid']
        )
        test_dataloader = DataLoader(dataset=test, shuffle=False,
                                     batch_size=64)

        model = getattr(
            torchvision.models,
            self.args['model_params']['model_architecture'])(
            pretrained=self.args['model_params']['use_pretrained_model'],
            progress=False)
        model = Classifier(
            model=model,
            truncate_to_layer=self.args['model_params']['truncate_to_layer'],
            classifier_cfg=self.args['model_params']['classifier_cfg'],
            final_activation_map_spatial_dimensions=self.args['model_params']['final_activation_map_spatial_dimensions']
            )
        _, inference_res = inference(
            model=model,
            test_loader=test_dataloader,
            has_labels=False,
            checkpoint_path=str(self.args['model_params']['checkpoint_path']))
        inference_res['experiment_id'] = self.args['experiment_id']

        inference_res.to_csv(Path(self.args['out_dir']) /
                             f'{self.args["experiment_id"]}_inference.csv',
                             index=False)

if __name__ == "__main__":
    inference = InferenceModule()
    inference.run()
