import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader as TorchDataLoader


class KfoldDataLoader:
    def __init__(self, train_dataset, y, n_splits, batch_size, shuffle=True, random_state=None):
        self.train_dataset = train_dataset
        self.y = y
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in skf.split(np.zeros(len(self.train_dataset)), self.y):
            train = Subset(dataset=self.train_dataset, indices=train_index)
            valid = Subset(dataset=self.train_dataset, indices=test_index)
            train_loader = TorchDataLoader(train, batch_size=self.batch_size, shuffle=True)
            valid_loader = TorchDataLoader(valid, batch_size=self.batch_size)
            yield train_loader, valid_loader
