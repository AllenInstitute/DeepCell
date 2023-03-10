import json
from pathlib import Path
from typing import List

import argschema
import torchvision
from torch.utils.data import DataLoader

from deepcell.cli.schemas.inference import InferenceSchema
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput
from deepcell.inference import inference, cv_performance
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.models.classifier import Classifier


class InferenceModule(argschema.ArgSchemaParser):
    """
    Runs inference for experiment and produces csv of predictions
    """
    default_schema = InferenceSchema

    def run(self):
        if len(self.args['model_inputs_paths']) > 1 and \
                self.args['mode'] != 'CV':
            raise RuntimeError('Passing multiple model_inputs_paths when '
                               'mode != "CV" is not understood')
        model_inputs = self._load_model_inputs()

        test_transform = RoiDataset.get_default_transforms(
            crop_size=self.args['data_params']['crop_size'], is_train=False,
            means=self.args['data_params']['channel_wise_means'],
            stds=self.args['data_params']['channel_wise_stds']
        )

        model = getattr(
            torchvision.models,
            self.args['model_params']['model_architecture'])(
            pretrained=self.args['model_params']['use_pretrained_model'],
            progress=False)
        model = Classifier(
            model=model,
            truncate_to_layer=self.args['model_params']['truncate_to_layer'],
            classifier_cfg=self.args['model_params']['classifier_cfg'])

        if self.args['mode'] in ('test', 'production'):
            model_inputs = model_inputs[0]

            test = RoiDataset(
                model_inputs=model_inputs,
                transform=test_transform
            )
            test_dataloader = DataLoader(dataset=test, shuffle=False,
                                         batch_size=self.args['batch_size'])

            _, inference_res = inference(
                model=model,
                test_loader=test_dataloader,
                has_labels=self.args['mode'] == 'test',
                checkpoint_path=str(self.args['model_load_path']))
        else:
            if len(model_inputs) > 1:
                # Just passing the raw split validation model inputs
                data_splitter = None
            else:
                data_splitter = DataSplitter(
                    model_inputs=model_inputs[0],
                    test_transform=test_transform,
                    seed=1234
                )
            inference_res, _ = cv_performance(
                model=model,
                model_inputs=model_inputs,
                data_splitter=data_splitter,
                checkpoint_path=self.args['model_load_path'],
                test_transform=test_transform)

        if self.args['mode'] == 'CV':
            out_filename = 'cv_preds.csv'
        elif self.args['mode'] == 'test':
            out_filename = 'test_preds.csv'
        elif self.args['experiment_id'] is not None:
            out_filename = f'{self.args["experiment_id"]}_inference.csv'
        else:
            out_filename = 'preds.csv'

        inference_res.to_csv(Path(self.args['save_path']) / f'{out_filename}',
                             index=False)

    def _load_model_inputs(self) -> List[List[ModelInput]]:
        res = []
        for path in self.args['model_inputs_paths']:
            with open(path, 'r') as f:
                model_inputs = json.load(f)
            model_inputs = [ModelInput(**model_input)
                            for model_input in model_inputs]
            res.append(model_inputs)
        return res


def main():
    inference = InferenceModule()
    inference.run()


if __name__ == "__main__":
    main()
