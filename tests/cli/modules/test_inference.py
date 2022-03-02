import json
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torchvision

from deepcell.cli.modules.inference import InferenceModule
from tests.util.util import get_test_data


class TestInferenceCli:

    @classmethod
    def setup_class(cls):

        data_dir = tempfile.TemporaryDirectory()
        dataset = get_test_data(write_dir=data_dir.name, is_train=False)

        net = torchvision.models.vgg11_bn(pretrained=True, progress=False)
        net.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 1)
        )
        checkpoint_path = tempfile.TemporaryDirectory()

        for fold in range(5):
            torch.save({'state_dict': net.state_dict()},
                       Path(checkpoint_path.name) / f'model_{fold}.pt')

        cls.data_dir = data_dir
        cls.dataset = dataset
        cls.checkpoint_path = checkpoint_path
        cls.experiment_id = '0'

    def teardown_class(self):
        self.data_dir.cleanup()
        self.checkpoint_path.cleanup()

    def test_inference_cli(self):
        with tempfile.TemporaryDirectory() as out_dir:
            with open(Path(out_dir) / 'model_inputs.json', 'w') as f:
                json.dump(self.dataset, f)

            with open(Path(out_dir) / 'model_inputs.json', 'r') as f:
                input_json = {
                    'experiment_id': self.experiment_id,
                    'data_params': {
                        'model_inputs_path': f.name
                    },
                    'model_params': {
                        'checkpoint_path': self.checkpoint_path.name,
                        'classifier_cfg': []
                    },
                    'out_dir': out_dir
                }
                inference_mod = InferenceModule(input_data=input_json, args=[])
                inference_mod.run()

                df = pd.read_csv(Path(out_dir) /
                                 f'{self.experiment_id}_inference.csv')
                assert df.shape[0] == len(self.dataset)
