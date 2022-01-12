from typing import List

import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from imgaug import augmenters as iaa, BoundingBox, BoundingBoxesOnImage

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
                 include_only_mask=False,
                 mask_out_projections=False,
                 use_correlation_projection=False,
                 try_center_soma_in_frame=False,
                 target='classification_label'):
        """
        A dataset of segmentation masks as identified by Suite2p with
        binary label "cell" or "not cell"
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
            include_only_mask
                Whether to include only the mask
            mask_out_projections:
                Whether to mask out projections to only include pixels in
                the mask
            use_correlation_projection
                Whether to use correlation projection instead of avg projection
            try_center_soma_in_frame
                The classifier has poor performance with a soma that is not
                centered in frame. Find the mask centroid and use that to
                center in the frame. Note: does not guarantee that the cell
                is perfectly centered as a soma with a long dendrite will
                have the centroid shifted.
            target
                What target to use.
                Can be classification_label or bounding_box
        """
        super().__init__()

        self._model_inputs = model_inputs
        self._image_dim = image_dim
        self.transform = transform
        self._exclude_mask = exclude_mask
        self._include_only_mask = include_only_mask
        self._mask_out_projections = mask_out_projections
        self._y = self._construct_target(target_name=target)
        self._classification_label = self._construct_target(
            target_name='classification_label')
        self._use_correlation_projection = use_correlation_projection
        self._try_center_soma_in_frame = try_center_soma_in_frame
        self._target_name = target

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
        """Target variable. The classification label or bounding box"""
        return self._y

    @property
    def classification_label(self) -> np.ndarray:
        """Classification label"""
        return self._classification_label

    def __getitem__(self, index):
        obs = self._model_inputs[index]

        input = self._construct_input(obs=obs)
        target = self._y[index]

        if self._target_name == 'bounding_box':
            x, y, width, height = target
            bounding_boxes = BoundingBoxesOnImage([
                BoundingBox(
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height)
            ], shape=self._image_dim)
        else:
            bounding_boxes = None

        if self.transform:
            imgaug_seq = self.transform.all_transform[0]
            torchvision_seq = \
                transforms.Compose(self.transform.all_transform[1:])

            aug = imgaug_seq(image=input, bounding_boxes=bounding_boxes)
            if bounding_boxes is not None:
                image_aug, bbs_aug = aug
                new_bounding_box = bbs_aug.bounding_boxes[0]
                target = np.array([new_bounding_box.x1,
                                   new_bounding_box.y1,
                                   new_bounding_box.x2 - new_bounding_box.x1,
                                   new_bounding_box.y2 - new_bounding_box.y1])
            else:
                image_aug = aug[0]

            input = torchvision_seq(image_aug)

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
            A soma might not be centered in frame, which hurts
            performance. Find translation pixels to center soma.

            Note: this does not guarantee the soma is
            perfectly centered due to long dendrites which shift the
            centroid away from the soma.

            1. Find the largest disjoint mask (which presumably contains the
            soma)
            2. Find centroid of mask found in (1)
            3. Find translation pixels to center centroid in (2) in frame
            Args:
                mask:
                    Segmentation mask
            Returns:
                Returns the translation amount in pixels to center a soma in
                frame
            """
            def calc_contour_centroid(contour) -> np.ndarray:
                M = cv2.moments(contour)
                centroid = np.array(
                    [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])
                return centroid

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            mask_areas = np.array([cv2.contourArea(x) for x in contours])
            largest_mask_idx = np.argmax(mask_areas)
            largest_contour = contours[largest_mask_idx]
            largest_mask_centroid = calc_contour_centroid(
                contour=largest_contour)
            frame_center = np.array(self._image_dim) / 2
            diff_from_center = frame_center - largest_mask_centroid
            return diff_from_center

        def center_cell(x: np.ndarray, translate_px: np.ndarray):
            """
            Centers a cell in frame
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

        with open(obs.mask_path, 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask)

        if self._include_only_mask:
            # Repeating mask along all 3 channels to conform with pretrained
            # imagenet model
            res[:, :, 0] = mask
            res[:, :, 1] = mask
            res[:, :, 2] = mask
            return res

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

        if self._try_center_soma_in_frame:
            translation_px = find_translation_px(mask=mask)
            res = center_cell(x=res, translate_px=translation_px)

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

    def _construct_target(self, target_name):
        if target_name == 'classification_label':
            target = np.array([int(x.label == 'cell') for x in
                              self._model_inputs])
        elif target_name == 'bounding_box':
            target = np.array([[x.bounding_box['x'],
                                x.bounding_box['y'],
                                x.bounding_box['width'],
                                x.bounding_box['height']]
                               for x in self._model_inputs])

        else:
            raise ValueError(f'target_name {target_name} not supported')
        return target