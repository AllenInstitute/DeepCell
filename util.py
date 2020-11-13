import numpy as np
from typing import Union, Tuple

from scipy.sparse import coo_matrix


# From https://github.com/AllenInstitute/segmentation-labeling-app/blob/master/slapp/transforms/array_utils.py#L9
def content_boundary_2d(arr: Union[np.ndarray, coo_matrix]) -> Tuple:
    """
    Get the minimal row/column boundaries for content in a 2d array.
    Parameters
    ==========
    arr: A 2d np.ndarray or coo_matrix. Other formats are also possible
        (csc_matrix, etc.) as long as they have the toarray() method to
         convert them to np.ndarray.
    Returns
    =======
    4-tuple of row/column boundaries that define the minimal rectangle
    around nonzero array content. Note that the maximum side boundaries
    follow python indexing rules, so the value returned is the actual
    max index + 1:
        top_bound: smallest row index
        bot_bound: largest row index + 1
        left_bound: smallest column index
        right_bound: largest column index + 1
    """
    if isinstance(arr, coo_matrix):
        col = arr.col
        row = arr.row
    else:
        if not isinstance(arr, np.ndarray):
            arr = arr.toarray()
        row, col = np.nonzero(arr)
    if not row.size:
        return 0, 0, 0, 0
    left_bound = col.min()
    right_bound = col.max() + 1
    top_bound = row.min()
    bot_bound = row.max() + 1
    return top_bound, bot_bound, left_bound, right_bound


# From https://github.com/AllenInstitute/segmentation-labeling-app/blob/master/slapp/transforms/array_utils.py#L121
def crop_2d_array(arr: Union[np.ndarray, coo_matrix]) -> np.ndarray:
    """
    Crop a 2d array to a rectangle surrounding all nonzero elements.
    Parameters
    ==========
    arr: A 2d np.ndarray or coo_matrix. Other formats are also possible
        (csc_matrix, etc.) as long as they have the toarray() method to
         convert them to np.ndarray.
    Raises
    ======
    ValueError if all elements are nonzero.
    """
    boundaries = content_boundary_2d(arr)
    if not isinstance(arr, np.ndarray):
        arr = arr.toarray()
    if sum(boundaries) == 0:
        raise ValueError("Cannot crop an empty array, or an array where all "
                         "elements are zero.")
    top_bound, bot_bound, left_bound, right_bound = boundaries
    return arr[top_bound:bot_bound, left_bound:right_bound]