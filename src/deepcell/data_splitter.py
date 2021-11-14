import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from deepcell.datasets.roi_dataset import RoiDataset


class DataSplitter:
    def __init__(self, manifest_path, data_dir, train_transform=None, test_transform=None, seed=None,
                 cre_line=None, exclude_mask=False, image_dim=(128, 128)):
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed
        self.cre_line = cre_line
        self.exlude_mask = exclude_mask
        self.image_dim = image_dim

    def get_train_test_split(self, test_size):
        full_dataset = RoiDataset(manifest_path=self.manifest_path,
                                  data_dir=self.data_dir, cre_line=self.cre_line, image_dim=self.image_dim)
        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=self.seed)
        train_index, test_index = next(sss.split(np.zeros(len(full_dataset)), full_dataset._y))

        roi_ids = np.array([x['roi-id'] for x in full_dataset.manifest])
        train_roi_ids = roi_ids[train_index]
        test_roi_ids = roi_ids[test_index]

        train_dataset = RoiDataset(roi_ids=train_roi_ids, manifest_path=self.manifest_path,
                                   transform=self.train_transform,
                                   data_dir=self.data_dir, exclude_mask=self.exlude_mask, image_dim=self.image_dim)
        test_dataset = RoiDataset(roi_ids=test_roi_ids, manifest_path=self.manifest_path,
                                  transform=self.test_transform, data_dir=self.data_dir,
                                  exclude_mask=self.exlude_mask, image_dim=self.image_dim)

        return train_dataset, test_dataset

    def get_cross_val_split(self, train_dataset, n_splits=5, shuffle=True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed)
        for train_index, test_index in skf.split(np.zeros(len(train_dataset)), train_dataset._y):
            roi_ids = np.array([x['roi-id'] for x in train_dataset.manifest])
            train_roi_ids = roi_ids[train_index]
            valid_roi_ids = roi_ids[test_index]

            train = RoiDataset(roi_ids=train_roi_ids, manifest_path=self.manifest_path,
                               transform=self.train_transform, data_dir=self.data_dir,
                               exclude_mask=self.exlude_mask, image_dim=self.image_dim)
            valid = RoiDataset(roi_ids=valid_roi_ids, manifest_path=self.manifest_path,
                               transform=self.test_transform, data_dir=self.data_dir,
                               exclude_mask=self.exlude_mask, image_dim=self.image_dim)
            yield train, valid


if __name__ == '__main__':
    manifest_path = '/Users/adam.amster/output.manifest'

    data_splitter = DataSplitter(manifest_path=manifest_path, seed=1234, data_dir='./data')
    train, test = data_splitter.get_train_test_split(test_size=.3)
