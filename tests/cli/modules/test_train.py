import random
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from deepcell.cli.modules.train import TrainModule
from deepcell.datasets.model_input import ModelInput


def test_train_cli(monkeypatch):
    labels = [random.choice(['cell', 'not cell'])] * 10
    dataset = []
    with tempfile.TemporaryDirectory() as d:
        for i in range(10):
            img = np.random.randint(low=0, high=255, size=(128, 128),
                                    dtype='uint8')
            with open(Path(d) / f'corr_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)
            with open(Path(d) / f'mask_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)
            with open(Path(d) / f'max_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)
            with open(Path(d) / f'avg_0_{i}.png', 'wb') as f:
                Image.fromarray(img).save(f)
            dataset.append(
                ModelInput(
                    experiment_id='0',
                    roi_id=str(i),
                    mask_path=Path(d) / f'mask_0_{i}.png',
                    correlation_projection_path=Path(d) / f'corr_0_{i}.png',
                    max_projection_path=Path(d) / f'max_0_{i}.png',
                    label=labels[i]
                )
            )

        with mock.patch('deepcell.cli.modules.train.'
                        'VisualBehaviorExtendedDataset',
                        autospec=True) as mock_vbds:
            mock_vbds.return_value.dataset = dataset
            with tempfile.TemporaryDirectory() as temp_path:
                input_json = {
                    'data_params': {
                        'download_path': temp_path
                    },
                    'save_path': temp_path,
                    'optimization_params': {
                        'n_epochs': 3
                    },
                    'test_fraction': 0.5,
                    'n_folds': 2
                }
                train_mod = TrainModule(input_data=input_json, args=[])
                train_mod.run()

                for fold in (0, 1):
                    checkpoint_path = Path(temp_path) / f'{fold}_model.pt'
                    assert checkpoint_path.exists()
