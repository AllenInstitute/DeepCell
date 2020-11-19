import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from Subset import Subset


class KfoldDataLoader:
    def __init__(self, train_dataset, y, n_splits, batch_size, shuffle=True, additional_train_transform=None,
                 random_state=None, crop_to_center=False, random_erasing=False):
        self.train_dataset = train_dataset
        self.y = y
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.additional_train_transform = additional_train_transform
        self.random_state = random_state
        self.center_crop = crop_to_center
        self.random_erasing = random_erasing

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for train_index, test_index in skf.split(np.zeros(len(self.train_dataset)), self.y):
            train = Subset(dataset=self.train_dataset, indices=train_index,
                           additional_transform=self.additional_train_transform,
                           apply_transform=True, center_crop=self.center_crop, apply_random_erasing=self.random_erasing)
            valid = Subset(dataset=self.train_dataset, indices=test_index, apply_transform=True,
                           center_crop=self.center_crop)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid, batch_size=self.batch_size)
            yield train_loader, valid_loader
