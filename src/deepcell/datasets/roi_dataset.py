from pathlib import Path
from typing import List

from PIL import Image
import numpy as np
from torch.utils.data import Dataset

from deepcell.datasets.model_input import ModelInput
from deepcell.transform import Transform
from deepcell.util import get_experiment_genotype_map


class RoiDataset(Dataset):
    def __init__(self,
                 dataset: List[ModelInput],
                 image_dim=(128, 128),
                 transform: Transform = None,
                 debug=False,
                 cre_line=None,
                 exclude_mask=False):
        """

        Args:
            dataset:
                A set of records in this dataset
            image_dim:
                Dimensions of artifacts
            transform:
                Transform to apply
            debug:
                Whether in debug mode. This will limit the number of
                records in the dataset to only a single example from each class
            cre_line:
                Whether to limit to a cre_line
            exclude_mask:
                Whether to exclude the mask from input to the model
        """
        super().__init__()

        self._dataset = dataset
        self._image_dim = image_dim
        self.transform = transform
        self._exclude_mask = exclude_mask
        self._y = np.array([x.label == 'cell' for x in self._dataset])

        if cre_line:
            experiment_genotype_map = get_experiment_genotype_map()
            self._filter_by_cre_line(
                experiment_genotype_map=experiment_genotype_map,
                cre_line=cre_line)

        if debug:
            not_cell_idx = np.argwhere(self._y == 0)[0][0]
            cell_idx = np.argwhere(self._y == 1)[0][0]
            self._dataset = [
                self._dataset[not_cell_idx], self._dataset[cell_idx]]
            self._y = np.array([0, 1])

    @property
    def artifacts(self) -> List[ModelInput]:
        return self._dataset
    
    @property
    def y(self) -> np.ndarray:
        return self._y

    def __getitem__(self, index):
        obs = self._dataset[index]

        input = self._construct_input(obs=obs)

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

        # TODO collate_fn should be used instead
        target = self._y[index]

        return input, target

    def __len__(self):
        return len(self._dataset)

    def _construct_input(self, obs) -> np.ndarray:
        """
        Construct a single input

        Returns:
            A numpy array of type uint8 and shape *dim, 3
        """
        with open(obs.avg_projection_path, 'rb') as f:
            avg = Image.open(f)
            avg = np.array(avg)

        with open(obs.max_projection_path, 'rb') as f:
            max = Image.open(f)
            max = np.array(max)

        with open(obs.mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask)

        res = np.zeros((*self._image_dim, 3), dtype=np.uint8)
        res[:, :, 0] = avg
        res[:, :, 1] = max

        if self._exclude_mask:
            res[:, :, 2] = max
        else:
            try:
                res[:, :, 2] = mask
            except ValueError:
                # TODO fix this issue
                pass

        return res

    def _filter_by_cre_line(self, experiment_genotype_map, cre_line):
        """
        Modifies dataset by filtering by creline

        Args:
            experiment_genotype_map:
                Maps experiment to genotype
            cre_line:
                Cre_line to filter by
        Returns:
            None, modifies inplace
        """
        self._dataset = [
            x for x in self._dataset if experiment_genotype_map[
                x.experiment_id].startswith(cre_line)]


if __name__ == '__main__':
    def main():
        from deepcell.datasets.visual_behavior_dataset import VisualBehaviorDataset
        ds = VisualBehaviorDataset(debug=True,
                                   artifact_destination=Path('/tmp/artifacts'))
        dataset = RoiDataset(dataset=ds.dataset)
        dataset[0]
    main()