"""
A module to execute the python executable deepcell.cli.modules.train.
"""
import json
import logging
import os
import subprocess
import sys
import argparse
from pathlib import Path

from deepcell.cli.schemas.data import ModelInputSchema
from deepcell.cli.schemas.train import TrainSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# These are set by sagemaker
TRAINING_DATA_DIR = Path('/opt/ml/input/data/training')
TRAINING_OUTPUT_DIR = '/opt/ml/model'

INPUT_JSON_PATH = '/opt/ml/train_input.json'
MODEL_INPUTS_PATH = '/opt/ml/model_inputs.json'


def main():
    _update_paths_to_container_paths()

    cmd = f'python -m deepcell.cli.modules.train --input_json ' \
          f'{INPUT_JSON_PATH}'
    cmd = cmd.split(' ')
    process = subprocess.Popen(cmd, stdout=sys.stdout,
                               stderr=sys.stderr,
                               env=os.environ)
    process.communicate()

    if process.returncode != 0:
        # Cause the training job to fail
        sys.exit(255)


def _update_paths_to_container_paths():
    """Update paths from local paths to container paths.
    The paths in the input json are expected to be local paths, but we need
    them to be container paths.
    Returns
    --------
    None, updates paths and overwrites input jsons with updated paths
    """

    with open(INPUT_JSON_PATH) as f:
        train_cfg = json.load(f)

    with open(MODEL_INPUTS_PATH) as f:
        model_inputs = json.load(f)

    # Update model input paths
    for model_input in model_inputs:
        model_input['mask_path'] = (
            TRAINING_DATA_DIR / Path(model_input['mask_path']).name)
        model_input['max_projection_path'] = (
            TRAINING_DATA_DIR / Path(model_input['max_projection_path']).name)
        if 'avg_projection_path' in model_input:
            model_input['avg_projection_path'] = (
                TRAINING_DATA_DIR /
                Path(model_input['avg_projection_path']).name)
        if 'correlation_projection_path' in model_input:
            model_input['correlation_projection_path'] = (
                TRAINING_DATA_DIR /
                Path(model_input['correlation_projection_path']).name)

    with open(MODEL_INPUTS_PATH, 'w') as f:
        f.write(ModelInputSchema().dumps(model_inputs))

    # Update model inputs path
    train_cfg['data_params']['model_inputs_path'] = MODEL_INPUTS_PATH

    # Update output path
    train_cfg['save_path'] = TRAINING_OUTPUT_DIR

    # Write out updated input json
    with open(INPUT_JSON_PATH, 'w') as f:
        f.write(TrainSchema().dumps(train_cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'train',
        help='This is a dummy argument that doesn\'t do '
        'anything. It is required due to the way '
        'sagemaker runs the container. Sagemaker always passes an argument of '
        '"train". See https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-dockerfile.html')   # noqa E501
    args = parser.parse_args()

    main()
