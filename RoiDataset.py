import os

from PIL import Image
from croissant.utils import read_jsonlines
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from Transform import Transform
from util import get_experiment_genotype_map


class RoiDataset(Dataset):
    def __init__(self, manifest_path, project_name, data_dir, image_dim=(128, 128), roi_ids=None,
                 transform: Transform = None, debug=False, has_labels=True, parse_from_manifest=True,
                 cre_line=None, include_mask=False, video_max_frames=None):
        super().__init__()

        if not parse_from_manifest and roi_ids is None:
            raise ValueError('need to provide roi ids if not parsing from manifest')

        self.manifest_path = manifest_path
        self.project_name = project_name
        self.data_dir = data_dir
        self.image_dim = image_dim
        self.transform = transform
        self.has_labels = has_labels
        self.include_mask = include_mask
        self.video_max_frames = video_max_frames

        experiment_genotype_map = get_experiment_genotype_map()

        if parse_from_manifest:
            manifest = read_jsonlines(uri=self.manifest_path)
            self.manifest = [x for x in manifest]
        else:
            self.manifest = [{'roi-id': roi_id} for roi_id in roi_ids]

        if cre_line:
            self.manifest = self._filter_by_cre_line(experiment_genotype_map=experiment_genotype_map, cre_line=cre_line)

        if roi_ids is not None:
            self.manifest = [x for x in self.manifest if str(x['roi-id']) in set([str(x) for x in roi_ids])]
            self.roi_ids = [x['roi-id'] for x in self.manifest]
        else:
            self.roi_ids = [x['roi-id'] for x in self.manifest]

        self.y = self._get_labels() if self.has_labels else None

        if parse_from_manifest:
            self.cre_line = self._get_creline(experiment_genotype_map=experiment_genotype_map)
        else:
            self.cre_line = None

        if debug:
            not_cell_idx = np.argwhere(self.y == 0)[0][0]
            cell_idx = np.argwhere(self.y == 1)[0][0]
            self.manifest = [self.manifest[not_cell_idx], self.manifest[cell_idx]]
            self.y = np.array([0, 1])
            self.roi_ids = [x['roi-id'] for x in self.manifest]

    def __getitem__(self, index):
        obs = self.manifest[index]

        video = self._get_video(obs=obs)

        if self.transform:
            video = self.transform.all_transform(video)

        # Note if no labels, 0.0 is given as the target
        # TODO collate_fn should be used instead
        target = self.y[index] if self.has_labels else 0

        return video, target

    def __len__(self):
        return len(self.manifest)

    def _get_labels(self):
        labels = [x[self.project_name]['majorityLabel'] for x in self.manifest]
        labels = [int(x == 'cell') for x in labels]
        labels = np.array(labels)
        return labels

    def _get_creline(self, experiment_genotype_map):
        cre_lines = []
        for x in self.manifest:
            if 'cre_line' in x:
                cre_lines.append(x['cre_line'])
            else:
                cre_lines.append(experiment_genotype_map[x['experiment-id']][:3])
        return cre_lines

    def _get_video(self, obs):
        roi_id = obs['roi-id']

        data_dir = self.data_dir

        v = np.load(f'{data_dir}/video_{roi_id}.npy')

        if self.video_max_frames:
            # If video shorter than video_max_frames,
            # pad with zeros
            if v.shape[0] < self.video_max_frames:
                diff = self.video_max_frames - v.shape[0]
                pad = np.zeros((diff, *self.image_dim))
                v = np.concatenate([v, pad])
            else:
                v = v[:self.video_max_frames]

        if self.include_mask:
            v = self._add_mask(video=v, roi_id=roi_id)

        # Convert frames to 3 channels
        v = np.repeat(v[:, :, :, np.newaxis], 3, axis=3)

        return v

    def _add_mask(self, video, roi_id):
        data_dir = self.data_dir
        outline = Image.open(f'{data_dir}/outline_{roi_id}.png')
        outline = np.array(outline)
        outline_copy = outline.copy()
        outline_copy[outline == 0] = 255
        outline_copy[outline == 255] = 0
        outline = Image.fromarray(outline_copy)

        for i in range(video.shape[0]):
            frame = Image.fromarray(video[i])

            frame.paste(outline, box=(0, 0), mask=outline)
            frame = np.array(frame)
            video[i] = frame

        return video

    def _filter_by_cre_line(self, experiment_genotype_map, cre_line):
        filtered = [x for x in self.manifest if experiment_genotype_map[x['experiment-id']].startswith(cre_line)]
        return filtered


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


if __name__ == '__main__':
    project_name = 'ophys-experts-slc-oct-2020_ophys-experts-go-big-or-go-home'
    manifest_path = 's3://prod.slapp.alleninstitute.org/behavior_slc_oct_2020_behavior_3cre_1600roi_merged/output.manifest'

    dataset = RoiDataset(manifest_path=manifest_path, project_name=project_name, data_dir='./data')

