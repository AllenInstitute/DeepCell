import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader

from Subset import Subset as SubsetWithTransform


class KfoldDataLoader:
    def __init__(self, train_dataset, y, n_splits, batch_size, shuffle=True, random_state=None,
                 additional_train_transform=None):
        self.train_dataset = train_dataset
        self.y = y
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.additional_train_transform = additional_train_transform

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in skf.split(np.zeros(len(self.train_dataset)), self.y):
            train = Subset(dataset=self.train_dataset, indices=train_index)
            train = SubsetWithTransform(subset=train, additional_transforms=self.additional_train_transform)

            valid = Subset(dataset=self.train_dataset, indices=test_index)
            valid = SubsetWithTransform(subset=valid)

            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid, batch_size=self.batch_size)
            yield train_loader, valid_loader
