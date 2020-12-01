from PIL import Image
from croissant.utils import read_jsonlines
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from Transform import Transform


class SlcDataset(Dataset):
    def __init__(self, manifest_path, project_name, data_dir, image_dim=(128, 128), roi_ids=None,
                 transform: Transform = None, debug=False):
        super().__init__()

        self.manifest_path = manifest_path
        self.project_name = project_name
        self.data_dir = data_dir
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

        if self.transform:
            avg, max_, mask = input[:, :, 0], input[:, :, 1], input[:, :, 2]
            if self.transform.avg_transform:
                avg = self.transform.avg_transform(avg)
                input[:, :, 0] = avg
            if self.transform.max_transform:
                max_ = self.transform.max_transform(max_)
                input[:, :, 1] = max_
            if self.transform.mask_transform:
                mask = self.transform.mask_transform(mask)
                input[:, :, 2] = mask

            if self.transform.all_transform:
                input = self.transform.all_transform(input)

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

        data_dir = self.data_dir
        with open(f'{data_dir}/avg_{roi_id}.png', 'rb') as f:
            avg = Image.open(f)
            avg = np.array(avg)

        with open(f'{data_dir}/max_{roi_id}.png', 'rb') as f:
            max = Image.open(f)
            max = np.array(max)

        with open(f'{data_dir}/mask_{roi_id}.png', 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask)

        res = np.zeros((*self.image_dim, 3), dtype=np.uint8)
        res[:, :, 0] = avg
        res[:, :, 1] = max
        res[:, :, 2] = mask

        return res


class SlcSampler(Sampler):
    def __init__(self, y, cell_prob=0.2):
        self.positive_proba = cell_prob

        self.positive_idxs = np.where(y == 1)[0]
        self.negative_idxs = np.where(y == 0)[0]

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative

