import numpy as np
import pandas as pd

from RoiDataset import RoiDataset


def get_random_roi(data: RoiDataset, label):
    idxs = np.where(data.y == label)[0]
    idx = np.random.choice(idxs)
    roi_id = data.roi_ids[idx]

    return roi_id


def get_experiment_genotype_map():
    experiment_metadata = pd.read_csv('ophys_metadata_lookup.txt')
    experiment_metadata.columns = [c.strip() for c in experiment_metadata.columns]
    return experiment_metadata[['experiment_id', 'genotype']].set_index('experiment_id') \
        .to_dict()['genotype']