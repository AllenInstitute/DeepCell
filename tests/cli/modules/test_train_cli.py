import json
import tempfile
from pathlib import Path
import pytest

from deepcell.cli.modules.train import TrainRunner
from deepcell.testing.util import get_test_data


@pytest.mark.parametrize('fold', [None, 0])
def test_train_cli(monkeypatch, fold):
    with tempfile.TemporaryDirectory() as d:
        dataset = get_test_data(write_dir=d, exp_id='0')

        with tempfile.TemporaryDirectory() as temp_path:
            with open(Path(temp_path) / 'model_inputs.json', 'w') as f:
                json.dump(dataset, f)

            with open(Path(temp_path) / 'model_inputs.json', 'r') as f:
                input_json = {
                    'train_model_inputs_path': f.name,
                    'validation_model_inputs_path': f.name,
                    'save_path': temp_path,
                    'optimization_params': {
                        'n_epochs': 3
                    },
                    'fold': fold
                }
                train_mod = TrainRunner(input_data=input_json, args=[])
                train_mod.run()

                fp = 'model.pt' if fold is None else f'{fold}_model.pt'
                checkpoint_path = Path(temp_path) / fp
                assert checkpoint_path.exists()
