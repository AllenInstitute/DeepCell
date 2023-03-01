import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest
import torch
import torchvision

from deepcell.cli.modules.inference import InferenceModule
from deepcell.testing.util import get_test_data


class TestInferenceCli:

    @classmethod
    def setup_class(cls):

        data_dir = tempfile.TemporaryDirectory()
        train_dataset = get_test_data(
            write_dir=data_dir.name,
            is_train=True,
            exp_id='0'
        )
        prod_dataset = get_test_data(
            write_dir=data_dir.name,
            is_train=False,
            exp_id='0'
        )

        net = torchvision.models.vgg11_bn(pretrained=True, progress=False)
        net.classifier = torch.nn.Sequential(torch.nn.Linear(512, 1))
        checkpoint_path = tempfile.TemporaryDirectory()

        for fold in range(5):
            torch.save({'state_dict': net.state_dict()},
                       Path(checkpoint_path.name) / f'{fold}_model.pt')

        cls.data_dir = data_dir
        cls.prod_dataset = prod_dataset
        cls.train_dataset = train_dataset
        cls.checkpoint_path = checkpoint_path
        cls.experiment_id = '0'

    def teardown_class(self):
        self.data_dir.cleanup()
        self.checkpoint_path.cleanup()

    @pytest.mark.parametrize('mode', ('test', 'CV', 'production'))
    def test_inference_cli(self, mode):
        dataset = self.prod_dataset if mode == 'production' else \
            self.train_dataset
        with tempfile.TemporaryDirectory() as out_dir:
            with open(Path(out_dir) / 'model_inputs.json', 'w') as f:
                json.dump(dataset, f)

            with open(Path(out_dir) / 'model_inputs.json', 'r') as f:
                input_json = {
                    'experiment_id': self.experiment_id,
                    'model_inputs_paths': [f.name],
                    'model_load_path': self.checkpoint_path.name,
                    'mode': mode,
                    'model_params': {
                        'classifier_cfg': []
                    },
                    'save_path': out_dir
                }
                inference_mod = InferenceModule(input_data=input_json, args=[])
                inference_mod.run()

                if mode == 'CV':
                    filename = 'cv_preds.csv'
                elif mode == 'test':
                    filename = 'test_preds.csv'
                elif self.experiment_id is not None:
                    filename = f'{self.experiment_id}_inference.csv'
                else:
                    filename = 'preds.csv'
                df = pd.read_csv(Path(out_dir) / filename)
                assert df.shape[0] == len(dataset)
