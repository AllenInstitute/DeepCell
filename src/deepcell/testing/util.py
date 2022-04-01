import random
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image


def get_test_data(write_dir: str,
                  is_train: bool = True,
                  n_rois: int = 10,
                  exp_mod: int = 2) -> List[Dict]:
    """

    Parameters
    ----------
    write_dir
        Directory to write test data to
    is_train
        Whether these are train inputs
    n_rois
        number of test rois to create.
    exp_mod
        Mod value to use to create experiment ids.

    Returns
    -------
    dataframe of metadata
    """
    labels = [random.choice(['cell', 'not cell'])] * n_rois
    dataset = []
    for i in range(n_rois):
        img = np.random.randint(low=0, high=255, size=(128, 128),
                                dtype='uint8')
        with open(Path(write_dir) / f'corr_0_{i}.png', 'wb') as f:
            Image.fromarray(img).save(f)
        with open(Path(write_dir) / f'mask_0_{i}.png', 'wb') as f:
            Image.fromarray(img).save(f)
        with open(Path(write_dir) / f'max_0_{i}.png', 'wb') as f:
            Image.fromarray(img).save(f)
        with open(Path(write_dir) / f'avg_0_{i}.png', 'wb') as f:
            Image.fromarray(img).save(f)
        d = {
            'experiment_id': str(i % exp_mod),
            'roi_id': i,
            'mask_path': str(Path(write_dir) / f'mask_0_{i}.png'),
            'correlation_projection_path': str(Path(write_dir) / f'corr_0_{i}.png'),    # noqa E501
            'max_projection_path': str(Path(write_dir) / f'max_0_{i}.png'),
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
