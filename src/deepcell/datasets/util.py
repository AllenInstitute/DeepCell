import cv2
import numpy as np
import imgaug.augmenters as iaa


def calc_roi_centroid(
        image: np.ndarray, brightness_quantile=0.8,
        image_dimensions=(128, 128),
        use_mask=False,
        centroid_dist_from_center_outlier=12) -> np.ndarray:
    """
    Calculates ROI centroid. The pixels are weighted by pixel intensity in
    image if use_mask is False. Falls back to mask if intensities are 0 in
    masked region.
    Pixel intensities act as weights and since the soma should have higher
    intensity than the rest of the ROI, this should choose a point close to
    the center of the soma.

    Pixels brighter than brightness_quantile are given much higher
    weight in order to better calculate a centroid of the soma.
    Args:
        image:
            See `image` arg in `center_roi`
        brightness_quantile:
            Pixel brightness, below which will be zeroed out in centroid
            calculation, giving higher weight to brightest pixels
        image_dimensions
            Image dimensions
        use_mask
            Whether to use the mask to calculate the centroid.
            If False, then intensities will be used, but will still fall
            back to the mask if intensities are 0 in masked region.
        centroid_dist_from_center_outlier
            The distance the centroid needs to be from the center to be
            considered an outlier. The default of 12 was found by visually
            inspecting the distribution of distances from the centers of
            manually annotated bounding boxes around soma to the center of
            the frame


    Returns:
        x, y of centroid in image coordinates
    """
    def calculate_centroid(intensities: np.ndarray,
                           mask: np.ndarray, binary_image=False) -> np.ndarray:
        """Calculates centroid using intensities or falls back to mask if
        all pixels are 0 in masked region."""
        M = cv2.moments(intensities, binaryImage=binary_image)
        if M['m00'] == 0:
            # Intensities are all 0. Try again using mask
            M = cv2.moments(mask, binaryImage=True)
        centroid = np.array(
            [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])])
        return centroid

    def calc_is_outlier_centroid(intensities: np.ndarray, mask: np.ndarray,
                            binary_image=False):
        """Returns whether the centroid is an outlier distance away from
        center"""
        centroid = calculate_centroid(intensities=intensities, mask=mask,
                                      binary_image=binary_image)
        center = np.array(image_dimensions) / 2
        dist_from_center = np.sqrt(((centroid - center)**2).sum())
        return dist_from_center > centroid_dist_from_center_outlier

    mask = image[:, :, 2].copy()

    # intensities are correlation projection, or max projection if not
    # available
    intensities = image[:, :, 0].copy()

    # mask out intensities to isolate masked pixels for quantile calculation
    intensities[np.where(mask == 0)] = 0

    binary_image = False

    is_outlier = calc_is_outlier_centroid(intensities=intensities, mask=mask,
                                          binary_image=binary_image)

    if use_mask or not is_outlier or intensities.max() == 0:
        # Use the mask to calculate centroid
        intensities = mask
        binary_image = True
    else:
        # We are using the projection to calculate the centroid and
        # the centroid is an outlier distance away from center.
        # Zeroing out pixels less than brightness quantile was found to
        # push centroid closer to center of soma in cases where soma was
        # connected to a long dendrite.
        low = np.quantile(intensities[intensities.nonzero()],
                          brightness_quantile)
        intensities[intensities <= low] = 0
    centroid = calculate_centroid(intensities=intensities, mask=mask,
                                  binary_image=binary_image)
    return centroid


def center_roi(image: np.ndarray, image_dim=(128, 128),
               brightness_quantile=0.8,
               use_mask=False,
               centroid_dist_from_center_outlier=12) -> np.ndarray:
    """
    Centers an ROI in frame
    Args:
        image:
            The input image
            The first channel is expected to be the max projection
            The second channel is expected to be the correlation projection
            or average projection if the correlation projection is not
            available
            The third channel is expected to be the segmentation mask
        image_dim:
            image dimensions
        brightness_quantile
            See `brightness_quantile` arg in `calc_roi_centroid`
        use_mask
            See `use_mask` arg in `calc_roi_centroid`
        centroid_dist_from_center_outlier
            See `centroid_dist_from_center_outlier` arg in `calc_roi_centroid`
    Returns:
        Input translated so that centroid is in center of frame
    """
    centroid = calc_roi_centroid(
        image=image,
        brightness_quantile=brightness_quantile,
        image_dimensions=image_dim,
        use_mask=use_mask,
        centroid_dist_from_center_outlier=centroid_dist_from_center_outlier)

    frame_center = np.array(image_dim) / 2
    diff_from_center = frame_center - centroid

    seq = iaa.Sequential([
        iaa.Affine(translate_px={'x': int(diff_from_center[0]),
                                 'y': int(diff_from_center[1])})
    ])
    image = seq(image=image)
    return image
