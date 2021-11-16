from typing import List

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from deepcell.datasets.model_input import ModelInput
from deepcell.datasets.roi_dataset import RoiDataset
from deepcell.datasets.visual_behavior_dataset import VisualBehaviorDataset
from deepcell.transform import Transform


class DataSplitter:
    def __init__(self, artifacts: List[ModelInput], train_transform=None,
                 test_transform=None, seed=None,
                 cre_line=None, exclude_mask=False, image_dim=(128, 128)):
        self._artifacts = artifacts
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.seed = seed
        self.cre_line = cre_line
        self.exclude_mask = exclude_mask
        self.image_dim = image_dim

    def get_train_test_split(self, test_size):
        full_dataset = RoiDataset(dataset=self._artifacts,
                                  cre_line=self.cre_line,
                                  image_dim=self.image_dim)
        sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size,
                                     random_state=self.seed)
        train_index, test_index = next(sss.split(np.zeros(len(full_dataset)),
                                                 full_dataset.y))

        train_dataset = self._get_dataset(index=train_index,
                                          transform=self.train_transform)
        test_dataset = self._get_dataset(index=test_index,
                                         transform=self.test_transform)

        return train_dataset, test_dataset

    def get_cross_val_split(self, train_dataset: RoiDataset, n_splits=5,
                            shuffle=True):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                              random_state=self.seed)
        for train_index, test_index in skf.split(np.zeros(len(train_dataset)),
                                                 train_dataset.y):
            train = self._get_dataset(index=train_index,
                                      transform=self.train_transform)
            valid = self._get_dataset(index=test_index,
                                      transform=self.test_transform)
            yield train, valid

    def _get_dataset(self, index: List[int],
                     transform: Transform) -> RoiDataset:
        """Returns RoiDataset of Artifacts at index

        Args:
            index:
                List of index of artifacts to construct dataset
            transform:
                transform to pass to RoiDataset

        Returns
            RoiDataset
        """
        index = set(index)
        artifacts = [x for i, x in enumerate(self._artifacts) if i in index]
        return RoiDataset(dataset=artifacts,
                          transform=transform,
                          exclude_mask=self.exclude_mask,
                          image_dim=self.image_dim)


if __name__ == '__main__':
    def main():
        from pathlib import Path
        dest = Path('/tmp/artifacts')
        dataset = VisualBehaviorDataset(artifact_destination=dest, debug=True)
        data_splitter = DataSplitter(artifacts=dataset.dataset, seed=1234)
        train, test = data_splitter.get_train_test_split(test_size=.3)

    main()
