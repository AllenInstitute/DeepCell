import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from SlcDataset import SlcDataset


class DataSplitter:
    def __init__(self, manifest_path, project_name, train_transform=None, test_transform=None, seed=None):
        self.manifest_path = manifest_path
        self.project_name = project_name
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed

    def get_train_test_split(self, test_size):
        full_dataset = SlcDataset(manifest_path=self.manifest_path, project_name=self.project_name)
        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=self.seed)
        train_index, test_index = next(sss.split(np.zeros(len(full_dataset)), full_dataset.y))

        roi_ids = np.array([x['roi-id'] for x in full_dataset.manifest])
        train_roi_ids = roi_ids[train_index]
        test_roi_ids = roi_ids[test_index]

        train_dataset = SlcDataset(roi_ids=train_roi_ids, manifest_path=self.manifest_path,
                                   project_name=self.project_name, transform=self.train_transform)
        test_dataset = SlcDataset(roi_ids=test_roi_ids, manifest_path=self.manifest_path,
                                  project_name=self.project_name, transform=self.test_transform)

        return train_dataset, test_dataset

    def get_cross_val_split(self, train_dataset, n_splits=5, shuffle=True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed)
        for train_index, test_index in skf.split(np.zeros(len(train_dataset)), train_dataset.y):
            roi_ids = np.array([x['roi-id'] for x in train_dataset.manifest])
            train_roi_ids = roi_ids[train_index]
            valid_roi_ids = roi_ids[test_index]

            train = SlcDataset(roi_ids=train_roi_ids, manifest_path=self.manifest_path,
                               project_name=self.project_name, transform=self.train_transform)
            valid = SlcDataset(roi_ids=valid_roi_ids, manifest_path=self.manifest_path,
                               project_name=self.project_name, transform=self.test_transform)
            yield train, valid