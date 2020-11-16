from torch.utils.data import Dataset
from torchvision import transforms


class Subset(Dataset):
    """
    Subclass of Dataset which augments pytorch Subset with transforms
    """
    def __init__(self, subset, additional_transforms=None):
        self.subset = subset

        default_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        if additional_transforms:
            transform = additional_transforms + default_transform
        else:
            transform = default_transform

        transform = transforms.Compose(transform)

        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)