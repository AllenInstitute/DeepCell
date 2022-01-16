from typing import Optional, Tuple

import cv2
import numpy as np
import imgaug.augmenters as iaa


def calc_roi_centroid(image: np.ndarray, brightness_quantile=0.8) -> \
        np.ndarray:
    """
    Calculates ROI centroid weighted by pixel intensity in image, or falls
    back to mask if intensities are 0 in masked region. Pixel intensities
    act as weights and since the soma should have higher intensity than the
    rest of the ROI, this should choose a point close to the center of the
    soma.

    Pixels brighter than brightness_quantile are given much higher
    weight in order to better calculate a centroid of the soma.
    Args:
        image:
            Input image
        brightness_quantile:
            Pixel brightness, below which will be zeroed out in centroid
            calculation, giving higher weight to brightest pixels

    Returns:
        x, y of centroid in image coordinates
    """
    mask = image[:, :, 2]

    # intensities are correlation projection, or max projection if not
    # available
    intensities = image[:, :, 0].copy()

    # mask out intensities to isolate masked pixels for quantile calculation
    intensities[np.where(mask == 0)] = 0

    binary_image = False

    if intensities.max() == 0:
        # just use the mask
        intensities = mask
        binary_image = True
    else:
        low = np.quantile(intensities[intensities.nonzero()],
                          brightness_quantile)
        intensities[intensities <= low] = 0

    M = cv2.moments(intensities, binaryImage=binary_image)
    if M['m00'] == 0:
        # try again using mask
        M = cv2.moments(mask, binaryImage=True)
    centroid = np.array(
        [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])
    return centroid


def center_roi(x: np.ndarray, image_dim=(128, 128),
               brightness_quantile=0.8) -> np.ndarray:
    """
    Centers an ROI in frame
    Args:
        x:
            input
        image_dim:
            image dimensions
        brightness_quantile
            See `calc_roi_centroid`
    Returns:
        Input translated so that centroid is in center of frame
    """
    centroid = calc_roi_centroid(image=x,
                                 brightness_quantile=brightness_quantile)

    frame_center = np.array(image_dim) / 2
    diff_from_center = frame_center - centroid

    seq = iaa.Sequential([
        iaa.Affine(translate_px={'x': int(diff_from_center[0]),
                                 'y': int(diff_from_center[1])})
    ])
    x = seq(image=x)
    return x
