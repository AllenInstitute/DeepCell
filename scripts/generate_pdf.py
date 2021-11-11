import matplotlib.figure as mplt_fig
import json
import PIL.Image
import copy
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


def filter_rois(raw_roi_list,
                stat_name=None,
                min_stat=None,
                min_area=None):

    output = []
    for roi in raw_roi_list:
        scores = roi['classifier_scores']
        if scores['area'] < min_area:
            continue
        stat = max(scores[f'corr_{stat_name}'],
                   scores[f'maximg_{stat_name}'],
                   scores[f'avgimg_{stat_name}'])
        if stat < min_stat:
            continue
        new_roi = copy.deepcopy(roi)
        if 'mask_matrix' in new_roi:
            new_roi['mask'] = new_roi['mask_matrix']
        if 'valid_roi' in new_roi:
            new_roi['valid'] = new_roi['valid_roi']
        if 'roi_id' in new_roi:
            new_roi['id'] = new_roi['roi_id']
        output.append(roi)
    return output


def generate_page(
        raw_roi_list=None,
        predictions: pd.DataFrame = None,
        color_map=None,
        corr_img=None,
        max_img=None,
        experiment_id=None,
        fontsize=15):

    n_rows = 2
    n_cols = 5

    raw_extract_list = []
    for roi in raw_roi_list:
        if 'mask_matrix' in roi:
            roi['mask'] = roi['mask_matrix']
        if 'valid_roi' in roi:
            roi['valid'] = roi['valid_roi']
        if 'roi_id' in roi:
            roi['id'] = roi['roi_id']
        raw_extract_list.append(roi)

    thresholds = (0.3, 0.4, 0.5)
    pred_ids = [predictions[predictions['y_score'] > threshold][
        'roi-id'].tolist() for threshold in thresholds]

    roi_id_roi_map = {x['id']: x for x in raw_extract_list}

    valid_roi_list = []
    for i in range(len(thresholds)):
        valid_roi_list.append([roi_id_roi_map[int(id.split('_')[-1])] for id in
                           pred_ids[i]])

    background_list = [corr_img]*n_cols
    background_list += [max_img]*n_cols

    title_list = [f'{experiment_id} correlation projection']
    title_list += [f'all ROIs ({len(raw_extract_list)})']
    title_list += [f'classifier score > {threshold} ({len(valid_roi_list[i])})'
                   for i, threshold in enumerate(thresholds)]

    title_list += [f'{experiment_id} max projection']
    title_list += [None]*(n_cols-1)

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


def path_to_rgb(file_path):
    img = np.array(PIL.Image.open(file_path, 'r'))
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
    parser.add_argument('--other_projections_path',
                        help='Path to max projection')
    parser.add_argument('--correlation_projection_path',
                        help='path to correlation projection images')
    parser.add_argument('--predictions_root', required=True)
    parser.add_argument('--n_roi', type=int, default=None)

    args = parser.parse_args()
    assert args.output_root is not None
    output_path = f'{args.output_root}/{args.pdf_name}.pdf'

    raw_dir = pathlib.Path(args.rois_path)
    assert raw_dir.is_dir()

    corr_dir = pathlib.Path(args.correlation_projection_path)
    assert corr_dir.is_dir()

    max_dir = pathlib.Path(args.other_projections_path)
    assert max_dir.is_dir()

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

            max_img_path = max_dir / f'{exp_id}_max_proj.png'
            if not max_img_path.is_file():
                raise RuntimeError(f'{max_img_path} is not file')
            corr_img_path = corr_dir / f'{exp_id}_correlation_proj.png'
            if not corr_img_path.is_file():
                raise RuntimeError(f'{corr_img_path} is not file')

            max_img = path_to_rgb(max_img_path)
            corr_img = path_to_rgb(corr_img_path)
            k = f'{exp_id}_suite2p_rois.json'
            color_map = global_color_map[k]
            predictions = pd.read_csv(predictions_path)
            with open(rois_path) as f:
                raw_rois_list = json.load(f)

            fig = generate_page(
                    raw_roi_list=raw_rois_list,
                    predictions=predictions,
                    color_map=color_map,
                    corr_img=corr_img,
                    max_img=max_img,
                    experiment_id=exp_id,
                    fontsize=10)
            pdf_handle.savefig(fig)
            ct += 1
            print(f'{ct} in {time.time()-t0}')
    print(f'wrote {output_path}')
