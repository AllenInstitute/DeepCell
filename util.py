import numpy as np

from SlcDataset import SlcDataset


def get_random_roi(data: SlcDataset, label):
    idxs = np.where(data.y == label)[0]
    idx = np.random.choice(idxs)
    roi_id = data.roi_ids[idx]

    return roi_id