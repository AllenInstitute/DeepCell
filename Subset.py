from typing import Sequence

from torch.utils.data import Dataset
from torchvision import transforms


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: Dataset, indices: Sequence[int], additional_transform=None,
                 apply_transform=False) -> None:
        self.dataset = dataset
        self.indices = indices
        self.apply_transform = apply_transform

        if additional_transform:
            transform = [
                transforms.ToTensor(),
                *additional_transform,
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        else:
            transform = [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        transform = transforms.Compose(transform)

        self.transform = transform

    def __getitem__(self, idx):
        sample, target = self.dataset[self.indices[idx]]

        if self.apply_transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.indices)