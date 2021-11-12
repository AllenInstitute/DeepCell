import h5py
import matplotlib.figure as mplt_fig
import json
import PIL.Image
import re
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_roi_contour_to_img)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    extract_roi_to_ophys_roi)

import argparse
import pathlib
import time


def plot_rois(extract_list, background_img, color_map, axis):
    for roi in extract_list:
        ophys = extract_roi_to_ophys_roi(roi)
        background_img = add_roi_contour_to_img(
                    background_img,
                    ophys,
                    color_map[str(ophys.roi_id)],
                    1.0)
    axis.imshow(background_img)
    return axis


def generate_page(
        raw_roi_list=None,
        predictions: pd.DataFrame = None,
        color_map=None,
        corr_img_noisy=None,
        max_img_noisy=None,
        corr_img_denoised=None,
        max_img_denoised=None,
        experiment_id=None,
        fontsize=15):

    n_rows = 4
    n_cols = 4

    raw_extract_list = []
    for roi in raw_roi_list:
        if 'mask_matrix' in roi:
            roi['mask'] = roi['mask_matrix']
        if 'valid_roi' in roi:
            roi['valid'] = roi['valid_roi']
        if 'roi_id' in roi:
            roi['id'] = roi['roi_id']
        raw_extract_list.append(roi)

    thresholds = (0.3, 0.5)
    pred_ids = [predictions[predictions['y_score'] > threshold][
        'roi-id'].tolist() for threshold in thresholds]

    roi_id_roi_map = {x['id']: x for x in raw_extract_list}

    valid_roi_list = []
    for i in range(len(thresholds)):
        valid_roi_list.append([roi_id_roi_map[int(id.split('_')[-1])] for id in
                           pred_ids[i]])

    background_list = [corr_img_denoised]*n_cols
    background_list += [corr_img_noisy]*n_cols
    background_list += [max_img_denoised]*n_cols
    background_list += [max_img_noisy]*n_cols

    title_list = []
    for title in ('(denoised)', '(noisy)'):
        title_list += [f'{experiment_id} correlation projection {title}']
        title_list += [f'all ROIs ({len(raw_extract_list)})']
        title_list += [f'classifier score > {threshold} ({len(valid_roi_list[i])})'
                       for i, threshold in enumerate(thresholds)]

    for title in ('(denoised)', '(noisy)'):
        title_list += [f'{experiment_id} max projection {title}']
        title_list += [f'all ROIs ({len(raw_extract_list)})']
        title_list += [f'classifier score > {threshold} ({len(valid_roi_list[i])})'
                       for i, threshold in enumerate(thresholds)]

    roi_set_List = [None, raw_extract_list]
    for i in range(len(thresholds)):
        roi_set_List.append(valid_roi_list[i])
    roi_set_List *= n_rows

    fig = mplt_fig.Figure(figsize=(5*n_cols, 5*n_rows))
    axis_list = [fig.add_subplot(n_rows, n_cols, ii+1)
                 for ii in range(n_rows*n_cols)]

    for i_axis in range(len(axis_list)):
        background = np.copy(background_list[i_axis])
        title = title_list[i_axis]
        roi_set = roi_set_List[i_axis]
        axis = axis_list[i_axis]

        if roi_set is not None:
            plot_rois(roi_set, background, color_map, axis)
        else:
            axis.imshow(background)

        if title is not None:
            axis.set_title(title, fontsize=fontsize)

    fig.tight_layout()
    return fig


def path_to_rgb(img: np.ndarray):
    if len(img.shape) == 3:
        assert img.shape[2] >= 3
        return img[:, :, :3]
    else:
        out = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for ic in range(3):
            out[:, :, ic] = img
        return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_name', required=True)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--rois_path', help='path to rois')
    parser.add_argument('--max_projection_path_noisy',
                        help='Path to noisy max projection')
    parser.add_argument('--max_projection_path_denoised',
                        help='Path to denoised max projection')
    parser.add_argument('--correlation_projection_path_noisy',
                        help='path to noisy correlation projection images')
    parser.add_argument('--correlation_projection_path_denoised',
                        help='path to denoised correlation projection images')
    parser.add_argument('--predictions_root', required=True)
    parser.add_argument('--n_roi', type=int, default=None)

    args = parser.parse_args()
    assert args.output_root is not None
    output_path = f'{args.output_root}/{args.pdf_name}.pdf'

    raw_dir = pathlib.Path(args.rois_path)
    assert raw_dir.is_dir()

    corr_dir_noisy = pathlib.Path(args.correlation_projection_path_noisy)
    corr_dir_denoised = pathlib.Path(args.correlation_projection_path_denoised)
    assert corr_dir_noisy.is_dir()

    max_dir_noisy = pathlib.Path(args.max_projection_path_noisy)
    max_dir_denoised = pathlib.Path(args.max_projection_path_denoised)
    assert max_dir_noisy.is_dir()

    predictions_dir = pathlib.Path(args.predictions_root)

    predictions_list = [n for n in predictions_dir.rglob('*.csv')]

    color_map_path = raw_dir.parent / 'original_color_map.json'
    with open(color_map_path, 'rb') as in_file:
        global_color_map = json.load(in_file)

    exp_id_pattern = re.compile('[0-9]+')

    t0 = time.time()
    ct = 0

    with PdfPages(output_path, 'w') as pdf_handle:
        for predictions_path in predictions_list:
            exp_id = exp_id_pattern.findall(str(predictions_path.name))[0]
            rois_path = raw_dir / f'{exp_id}_suite2p_rois.json'

            with h5py.File(max_dir_noisy / f'{exp_id}_max_proj.h5', 'r') as f:
                max_img_noisy = f['max'][()]
                low, high = np.quantile(max_img_noisy, (.1, .99))
                max_img_noisy[max_img_noisy <= low] = low
                max_img_noisy[max_img_noisy >= high] = high
                max_img_noisy = (max_img_noisy - max_img_noisy.min()) / (max_img_noisy.max() -
                                                       max_img_noisy.min())
                max_img_noisy *= 255
                max_img_noisy = max_img_noisy.astype('uint8')
            max_img_path_denoised = max_dir_denoised / f'{exp_id}_max_proj.png'

            if not max_img_path_denoised.is_file():
                raise RuntimeError(f'{max_img_path_denoised} is not file')
            max_img_denoised = np.array(PIL.Image.open(
                max_img_path_denoised, 'r'))

            with h5py.File(corr_dir_noisy / f'{exp_id}_ff050_img.h5', 'r') as f:
                corr_img_noisy = f['data'][()]
                low, high = np.quantile(corr_img_noisy, (.1, .9))
                corr_img_noisy[corr_img_noisy <= low] = low
                corr_img_noisy[corr_img_noisy >= high] = high
                corr_img_noisy = (corr_img_noisy - corr_img_noisy.min()) / (corr_img_noisy.max()
                                                          - corr_img_noisy.min())
                corr_img_noisy *= 255
                corr_img_noisy = corr_img_noisy.astype('uint8')

            corr_img_path_denoised = corr_dir_denoised / \
                                     f'{exp_id}_correlation_proj.png'
            corr_img_denoised = np.array(PIL.Image.open(
                corr_img_path_denoised, 'r'))


            max_img_noisy = path_to_rgb(max_img_noisy)
            corr_img_noisy = path_to_rgb(corr_img_noisy)

            max_img_denoised = path_to_rgb(max_img_denoised)
            corr_img_denoised = path_to_rgb(corr_img_denoised)

            k = f'{exp_id}_suite2p_rois.json'
            color_map = global_color_map[k]
            predictions = pd.read_csv(predictions_path)
            with open(rois_path) as f:
                raw_rois_list = json.load(f)

            fig = generate_page(
                    raw_roi_list=raw_rois_list,
                    predictions=predictions,
                    color_map=color_map,
                    corr_img_noisy=corr_img_noisy,
                    max_img_noisy=max_img_noisy,
                    corr_img_denoised=corr_img_denoised,
                    max_img_denoised=max_img_denoised,
                    experiment_id=exp_id,
                    fontsize=10)
            pdf_handle.savefig(fig)
            ct += 1
            print(f'{ct} in {time.time()-t0}')
    print(f'wrote {output_path}')
