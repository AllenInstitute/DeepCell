import random
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image

from deepcell.datasets.channel import Channel, channel_filename_prefix_map


def get_test_data(write_dir: str,
                  exp_id: str,
                  is_train: bool = True,
                  n_rois: int = 10
                  ) -> List[Dict]:
    """

    Parameters
    ----------
    write_dir
        Directory to write test data to
    exp_id
        Experiment id
    is_train
        Whether these are train inputs
    n_rois
        number of test rois to create.

    Returns
    -------
    dataframe of metadata
    """
    labels = [random.choice(['cell', 'not cell'])] * n_rois
    dataset = []
    for i in range(n_rois):
        img = np.random.randint(low=0, high=255, size=(128, 128),
                                dtype='uint8')
        channel_path_map = {
            Channel.MASK.value: str(
                Path(write_dir) /
                f'{channel_filename_prefix_map[Channel.MASK]}_{exp_id}_'
                f'{i}.png'),
            Channel.MAX_PROJECTION.value: str(
                Path(write_dir) /
                f'{channel_filename_prefix_map[Channel.MAX_PROJECTION]}_'
                f'{exp_id}_'
                f'{i}.png'),
            Channel.CORRELATION_PROJECTION.value: str(
                Path(write_dir) /
                f'{channel_filename_prefix_map[Channel.CORRELATION_PROJECTION]}'    # noqa E402
                f'_{exp_id}_{i}.png'),
            Channel.AVG_PROJECTION.value: str(
                Path(write_dir) /
                f'{channel_filename_prefix_map[Channel.AVG_PROJECTION]}'
                f'_{exp_id}_{i}.png')
        }
        for channel, path in channel_path_map.items():
            with open(path, 'wb') as f:
                Image.fromarray(img).save(f)

        d = {
            'experiment_id': exp_id,
            'roi_id': i,
            'channel_path_map': channel_path_map,
            'channel_order': [
                Channel.MASK.value,
                Channel.MAX_PROJECTION.value,
                Channel.CORRELATION_PROJECTION.value
            ]
        }
        if is_train:
            d['label'] = labels[i]
        dataset.append(d)

    return dataset


def get_exp_test_data(n_experiments: int = 2) -> List[Dict]:
    """Create dummy experiments.

    Parameters
    ----------
    n_experments
        Number of unique experiments to create.

    Returns
    -------
    exp_meta
        List of experiment metadatas.
    """
    exp_meta = []
    for exp_id in range(n_experiments):
        exp_meta.append({'experiment_id': str(exp_id),
                         'imaging_depth': 0,
                         'equipment': '2P',
                         'problem_experiment': False})
    return exp_meta
