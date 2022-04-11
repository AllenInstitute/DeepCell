import random
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image


def get_test_data(write_dir: str, is_train=True) -> List[Dict]:
    """

    Parameters
    ----------
    write_dir
        Directory to write test data to
    is_train
        Whether these are train inputs

    Returns
    -------
    dataframe of metadata
    """
    labels = [random.choice(['cell', 'not cell'])] * 10
    dataset = []
    for i in range(10):
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
            'experiment_id': '0',
            'roi_id': i,
            'mask_path': str(Path(write_dir) / f'mask_0_{i}.png'),
            'correlation_projection_path': str(Path(write_dir) / f'corr_0_{i}.png'),    # noqa E501
            'max_projection_path': str(Path(write_dir) / f'max_0_{i}.png'),
        }
        if is_train:
            d['label'] = labels[i]
        dataset.append(d)

    return dataset
