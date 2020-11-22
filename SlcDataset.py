from PIL import Image
from croissant.utils import read_jsonlines
import numpy as np
from torch.utils.data import Dataset


class SlcDataset(Dataset):
    def __init__(self, manifest_path, project_name, image_dim=(128, 128), roi_ids=None, transform=None, debug=False):
        super().__init__()

        self.manifest_path = manifest_path
        self.project_name = project_name
        self.image_dim = image_dim
        self.transform = transform

        manifest = read_jsonlines(uri=self.manifest_path)
        self.manifest = [x for x in manifest]
        self.roi_ids = roi_ids if roi_ids is not None else [x['roi-id'] for x in self.manifest]
        self.manifest = [x for x in self.manifest if x['roi-id'] in set(self.roi_ids)]

        self.y = self._get_labels()

        if debug:
            not_cell_idx = np.argwhere(self.y == 0)[0][0]
            cell_idx = np.argwhere(self.y == 1)[0][0]
            self.manifest = [self.manifest[not_cell_idx], self.manifest[cell_idx]]
            self.y = np.array([0, 1])
            self.roi_ids = [x['roi-id'] for x in self.manifest]

    def __getitem__(self, index):
        obs = self.manifest[index]

        input = self._extract_channels(obs=obs)
        input = Image.fromarray(input)

        if self.transform:
            input = self.transform(input)

        target = self.y[index]

        return input, target

    def __len__(self):
        return len(self.manifest)

    def _get_labels(self):
        labels = [x[self.project_name]['majorityLabel'] for x in self.manifest]
        labels = [int(x == 'cell') for x in labels]
        labels = np.array(labels)
        return labels

    def _extract_channels(self, obs):
        roi_id = obs['roi-id']

        with open(f'data/avg_{roi_id}.png', 'rb') as f:
            avg = Image.open(f)
            avg = np.array(avg)

        with open(f'data/max_{roi_id}.png', 'rb') as f:
            max = Image.open(f)
            max = np.array(max)

        with open(f'data/mask_{roi_id}.png', 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask)

        res = np.zeros((*self.image_dim, 3), dtype=np.uint8)
        res[:, :, 0] = avg
        res[:, :, 1] = max
        res[:, :, 2] = mask

        return res

