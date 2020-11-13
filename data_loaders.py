import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader


def get_data_loaders(full_dataset, y, test_size, batch_size, seed=None, train_split=True):
    if not train_split:
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=True)
        return train_loader, None, None

    sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=seed)
    train_index, test_index = next(sss.split(np.zeros(len(y)), y))

    train_subset = Subset(dataset=full_dataset, indices=train_index)
    test_subset = Subset(dataset=full_dataset, indices=test_index)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size)

    return train_loader, test_loader
