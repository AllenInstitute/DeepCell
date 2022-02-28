import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from deepcell.cli.modules.inference import InferenceModule


class TestInferenceCli:

    @classmethod
    def setup_class(cls):
        inputs_dir = tempfile.TemporaryDirectory()
        for i in range(10):
            img = np.random.randint(low=0, high=255, size=(128, 128),
                                    dtype='uint8')
            with open(Path(inputs_dir.name) / f'corr_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)
            with open(Path(inputs_dir.name) / f'mask_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)
            with open(Path(inputs_dir.name) / f'max_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)
            with open(Path(inputs_dir.name) / f'avg_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)

        net = torchvision.models.vgg11_bn(pretrained=True, progress=False)
        net.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 1)
        )
        checkpoint_path = tempfile.TemporaryDirectory()

        for fold in range(5):
            torch.save({'state_dict': net.state_dict()},
                       Path(checkpoint_path.name) / f'model_{fold}.pt')

        rois_path = Path(inputs_dir.name) / 'rois.json'
        rois = [
                {'id': 0},
                {'id': 1}
            ]
        with open(rois_path, 'w') as f:
            f.write(json.dumps(rois))

        cls.inputs_dir = inputs_dir
        cls.checkpoint_path = checkpoint_path
        cls.rois_path = rois_path
        cls.rois = rois
        cls.experiment_id = '0'

    def teardown(self):
        self.inputs_dir.cleanup()
        self.checkpoint_path.cleanup()

    def test_inference_cli(self):
        with tempfile.TemporaryDirectory() as out_dir:
            input_json = {
                'experiment_id': self.experiment_id,
                'data_params': {
                    'data_dir': self.inputs_dir.name,
                    'rois_path': str(self.rois_path)
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
            assert df.shape[0] == len(self.rois)
