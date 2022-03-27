import json
import tempfile
from pathlib import Path


from deepcell.cli.modules.train import TrainRunner
from tests.util.util import get_test_data


def test_train_cli(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        dataset = get_test_data(write_dir=d)

        with tempfile.TemporaryDirectory() as temp_path:
            with open(Path(temp_path) / 'model_inputs.json', 'w') as f:
                json.dump(dataset, f)

            with open(Path(temp_path) / 'model_inputs.json', 'r') as f:
                input_json = {
                    'data_params': {
                        'model_inputs_path': f.name
                    },
                    'save_path': temp_path,
                    'optimization_params': {
                        'n_epochs': 3
                    },
                    'test_fraction': 0.5,
                    'n_folds': 2
                }
                train_mod = TrainRunner(input_data=input_json, args=[])
                train_mod.run()

                for fold in (0, 1):
                    checkpoint_path = Path(temp_path) / f'{fold}_model.pt'
                    assert checkpoint_path.exists()
