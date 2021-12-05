from typing import List

import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa

from deepcell.datasets.model_input import ModelInput
from deepcell.transform import Transform
from deepcell.util import get_experiment_genotype_map


class RoiDataset(Dataset):
    def __init__(self,
                 model_inputs: List[ModelInput],
                 image_dim=(128, 128),
                 transform: Transform = None,
                 debug=False,
                 cre_line=None,
                 exclude_mask=False,
                 mask_out_projections=False,
                 use_correlation_projection=False,
                 center_soma_in_disjoint_mask=False):
        """

        Args:
            model_inputs:
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
            mask_out_projections:
                Whether to mask out projections to only include pixels in
                the mask
            use_correlation_projection
                Whether to use correlation projection instead of avg projection
            center_soma_in_disjoint_mask
                If the segmentation mask contains disjoint regions, take the
                largest, which is usually the soma, and center it.
        """
        super().__init__()

        self._model_inputs = model_inputs
        self._image_dim = image_dim
        self.transform = transform
        self._exclude_mask = exclude_mask
        self._mask_out_projections = mask_out_projections
        self._y = np.array([int(x.label == 'cell') for x in self._model_inputs])
        self._use_correlation_projection = use_correlation_projection
        self._center_soma_in_disjoint_mask = center_soma_in_disjoint_mask

        if cre_line:
            experiment_genotype_map = get_experiment_genotype_map()
            self._filter_by_cre_line(
                experiment_genotype_map=experiment_genotype_map,
                cre_line=cre_line)

        if debug:
            not_cell_idx = np.argwhere(self._y == 0)[0][0]
            cell_idx = np.argwhere(self._y == 1)[0][0]
            self._model_inputs = [
                self._model_inputs[not_cell_idx], self._model_inputs[cell_idx]]
            self._y = np.array([0, 1])

    @property
    def model_inputs(self) -> List[ModelInput]:
        return self._model_inputs
    
    @property
    def y(self) -> np.ndarray:
        return self._y

    def __getitem__(self, index):
        obs = self._model_inputs[index]

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
        return len(self._model_inputs)

    def _construct_input(self, obs: ModelInput) -> np.ndarray:
        """
        Construct a single input

        Returns:
            A numpy array of type uint8 and shape *dim, 3
        """
        def find_translation_px(mask: np.ndarray) -> np.ndarray:
            """
            The segmentation mask can be disjoint. This can make the
            segmentation mask not centered in frame, which hurts
            performance. Select the disjoint
            region with the largest area (assumes this is the soma) and find
            translation amount in pixels to make it centered
            Args:
                mask:
                    Segmentation mask
            Returns:
                Returns the translation amount in pixels to center a soma (
                assuming the soma has the largest area in a disjoint mask)
            """
            def calc_contour_centroid(contour) -> np.ndarray:
                M = cv2.moments(contour)
                centroid = np.array(
                    [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])
                return centroid

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 1:
                mask_areas = np.array([cv2.contourArea(x) for x in contours])
                largest_mask_idx = np.argmax(mask_areas)
                largest_contour = contours[largest_mask_idx]
                largest_mask_centroid = calc_contour_centroid(
                    contour=largest_contour)
                frame_center = np.array(self._image_dim) / 2
                diff_from_center = frame_center - largest_mask_centroid
                return diff_from_center
            return np.array([0, 0])

        def center_soma(x: np.ndarray, translate_px: np.ndarray):
            """
            Centers a soma in frame
            Args:
                x: input
                translate_px: amount to translate the input in frame

            Returns:
                input with soma centered
            """
            transform = transforms.Compose([
                iaa.Sequential([
                    iaa.Affine(translate_px={'x': int(translate_px[0]),
                                             'y': int(translate_px[1])})
                ]).augment_image
            ])
            x = transform(x)
            return x

        res = np.zeros((*self._image_dim, 3), dtype=np.uint8)

        if self._use_correlation_projection:
            if obs.correlation_projection_path is not None:
                with open(obs.correlation_projection_path, 'rb') as f:
                    corr = Image.open(f)
                    corr = np.array(corr)
                    res[:, :, 0] = corr
            else:
                with open(obs.avg_projection_path, 'rb') as f:
                    avg = Image.open(f)
                    avg = np.array(avg)
                    res[:, :, 0] = avg
        else:
            with open(obs.avg_projection_path, 'rb') as f:
                avg = Image.open(f)
                avg = np.array(avg)
                res[:, :, 0] = avg

        with open(obs.max_projection_path, 'rb') as f:
            max = Image.open(f)
            max = np.array(max)
            res[:, :, 1] = max

        with open(obs.mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask)

            if self._exclude_mask:
                res[:, :, 2] = max
            else:
                try:
                    res[:, :, 2] = mask
                except ValueError:
                    # TODO fix this issue
                    pass

        if self._mask_out_projections:
            res[:, :, 0][np.where(mask == 0)] = 0
            res[:, :, 1][np.where(mask == 0)] = 0

        if self._center_soma_in_disjoint_mask:
            translation_px = find_translation_px(mask=mask)
            res = center_soma(x=res, translate_px=translation_px)

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
        self._model_inputs = [
            x for x in self._model_inputs if experiment_genotype_map[
                x.experiment_id].startswith(cre_line)]


if __name__ == '__main__':
    def main():
        from torchvision import transforms
        from imgaug import augmenters as iaa
        artifacts = [ModelInput.from_data_dir(
            data_dir='/tmp/artifacts',
            experiment_id='871196379', roi_id='1')]
        all_transform = transforms.Compose([
            iaa.Sequential([
                iaa.CenterCropToFixedSize(height=60, width=60)
            ]).augment_image
        ])

        test_transform = Transform(all_transform=all_transform)
        dataset = RoiDataset(model_inputs=artifacts, transform=test_transform)
        for _ in dataset:
            pass
    main()
