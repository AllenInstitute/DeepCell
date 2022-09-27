import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import boto3
import json

import cv2
import h5py
import numpy as np
import pandas as pd
import shutil
from cell_labeling_app.database.database import db
from cell_labeling_app.database.schemas import JobRegion
from cell_labeling_app.util.util import get_completed_regions, get_region, \
    _is_roi_within_region
from deepcell.cli.modules.create_dataset import construct_dataset, \
    VoteTallyingStrategy
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.exp_metadata import ExperimentMetadata
from flask import Flask
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from ophys_etl.utils.rois import sanitize_extract_roi_list, \
    extract_roi_to_ophys_roi
from tqdm import tqdm
import seaborn as sns


def get_test_experiments():
    s3 = boto3.client('s3')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with open(tmp / 'all_model_inputs.json', 'wb') as f:
            s3.download_fileobj('dev.deepcell.alleninstitute.org',
                                'input_data/all/model_inputs.json', f)
        with open(tmp / 'train_model_inputs.json', 'wb') as f:
            s3.download_fileobj('dev.deepcell.alleninstitute.org',
                                'input_data/train/0/model_inputs.json', f)
        with open(tmp / 'val_model_inputs.json', 'wb') as f:
            s3.download_fileobj('dev.deepcell.alleninstitute.org',
                                'input_data/validation/0/model_inputs.json', f)
        with open(tmp / 'raw_labels.csv', 'wb') as f:
            s3.download_fileobj(
                'ophys.alleninstitute.org',
                'ssf/segmentation/classifier/raw_labels/raw_labels.csv', f)
        with open(tmp / 'all_model_inputs.json') as f:
            train = json.load(f)

        train = set([x['experiment_id'] for x in train])

        all = pd.read_csv(tmp / 'raw_labels.csv', dtype={'experiment_id': str})
        test = set([x for x in all['experiment_id'] if x not in train])
        return test


def get_labeled_regions_for_experiments(exp_ids):
    exp_ids = set(exp_ids)
    completed_regions = [
        get_region(region_id=x) for x in get_completed_regions()]
    completed_regions = [x for x in completed_regions
                         if str(x.experiment_id) in exp_ids]
    return completed_regions


def _get_labeling_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = \
        f'sqlite:////allen/programs/mindscope/workgroups/surround/' \
        f'denoising_labeling_2022/labeling/cell_labeling_app/' \
        f'cell_labeling_app.db'
    app.config['ARTIFACT_DIR'] = '/allen/programs/mindscope/workgroups/surround/denoising_labeling_2022/labeling/artifacts/'
    app.config['PREDICTIONS_DIR'] = None
    app.config['LABELERS_REQUIRED_PER_REGION'] = 3

    db.init_app(app)
    return app.config, app.app_context()


def _get_rois_within_regions(
    exp_id: str,
    regions: List[JobRegion],
    threshold_scaling: Optional[int] = None,
    ground_truth=False,
    groundtruth_labels: Optional[pd.DataFrame] = None
) -> List[Dict]:
    """
    Gets all rois from `exp_ids` within `regions`
    @param exp_id:
    @param regions: labeled regions
    @param threshold_scaling: optional threshold scaling to retrieve rois for
    @param ground_truth: whether to retrieve ground truth rois
    @return:

    """
    if not ground_truth and threshold_scaling is None:
        raise ValueError('Provide threshold_scaling if not getting '
                         'ground truth rois')
    if ground_truth and groundtruth_labels is None:
        raise ValueError('Provide labels if returning groundtruth rois')

    if ground_truth:
        rois_path = (
            f'/allen/programs/mindscope/workgroups/surround/'
            f'denoising_labeling_2022/'
            f'segmentations/{exp_id}/'
            f'{exp_id}_rois.json')
    else:
        rois_path = (
            f'/allen/programs/mindscope/workgroups/surround/'
            f'denoising_labeling_2022/'
            f'suite2p_threshold_scaling_analysis/{exp_id}/'
            f'{exp_id}_th{threshold_scaling}_rois.json')

    with open(rois_path) as f:
        rois = json.load(f)
    rois = sanitize_extract_roi_list(rois)
    for roi in rois:
        roi['experiment_id'] = exp_id

    if ground_truth:
        groundtruth_rois = set(groundtruth_labels[
                                   groundtruth_labels['label'] == 'cell']
                               [['experiment_id', 'roi_id']]
                      .apply(tuple, axis=1).values)
        res = [
            x for x in rois
            if (x['experiment_id'], x['id'])
            in groundtruth_rois
        ]
    else:
        res = []
        for region in regions:
            for roi in rois:
                if _is_roi_within_region(roi=roi, region=region):
                    res.append(roi)
    return res


def get_rois_in_labeled_regions_for_exp_id(
    exp_id: str,
    regions: List[JobRegion],
    threshold_scaling: Optional[int] = None,
    ground_truth=False,
    groundtruth_labels: Optional[pd.DataFrame] = None
):
    exp_regions = [x for x in regions
                   if str(x.experiment_id) == exp_id]
    rois = _get_rois_within_regions(
        threshold_scaling=threshold_scaling,
        exp_id=exp_id,
        regions=exp_regions,
        ground_truth=ground_truth,
        groundtruth_labels=groundtruth_labels
    )
    return rois


def get_rois_in_labeled_regions(
    exp_ids: List[str],
    regions: List[JobRegion],
    threshold_scaling: Optional[int] = None,
    ground_truth=False,
    groundtruth_labels: Optional[pd.DataFrame] = None
):
    res = []
    for exp_id in exp_ids:
        rois = get_rois_in_labeled_regions_for_exp_id(
            exp_id=exp_id,
            regions=regions,
            threshold_scaling=threshold_scaling,
            ground_truth=ground_truth,
            groundtruth_labels=groundtruth_labels
        )
        res += rois
    return res


def write_slurm_script_for_artifact_creation(
        exp_ids: List[str],
        threshold_scaling: int,
        rois: List[Dict]
):
    base_path = Path('/allen/aibs/informatics/aamster/segmentation/'
                     'threshold_scaling_analysis/')
    thumbnail_path = base_path / 'classifier_thumbnails'
    slurm_scripts_dir = Path(thumbnail_path / 'slurm')
    out_dir = base_path / f'threshold_scaling_{threshold_scaling}'
    log_path = slurm_scripts_dir / 'logs'
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(base_path / 'exp_ids.txt', 'w') as f:
        for exp_id in exp_ids:
            f.write(exp_id)
            f.write('\n')

    input_json_dir = slurm_scripts_dir / \
        f'threshold_scaling_{threshold_scaling}'
    os.makedirs(input_json_dir, exist_ok=True)

    for exp_id in exp_ids:
        with open(input_json_dir /
                  f'exp_{exp_id}.json', 'w') as f:
            d = {
                'video_path': f'/allen/programs/mindscope/workgroups/surround/denoising_labeling_2022/denoised_movies/{exp_id}/{exp_id}_denoised_video.h5',
                'roi_path': f'/allen/programs/mindscope/workgroups/surround/'
                            f'denoising_labeling_2022/'
                            f'suite2p_threshold_scaling_analysis/{exp_id}/'
                            f'{exp_id}_th{threshold_scaling}_rois.json',
                'graph_path': f'/allen/programs/mindscope/workgroups/surround/denoising_labeling_2022/denoised_movies/{exp_id}/{exp_id}_correlation_graph.pkl',
                'selected_rois': [x['id'] for x in rois
                                  if x['experiment_id'] == exp_id],
                'out_dir': str(out_dir)
            }
            f.write(json.dumps(d, indent=2))

    exp_id = '${exp_ids[$SLURM_ARRAY_TASK_ID]}'

    cmd = f'''
        /allen/aibs/informatics/aamster/miniconda3/bin/python -m \
        ophys_etl.modules.roi_cell_classifier.compute_classifier_artifacts \
        --input_json {input_json_dir}/exp_{exp_id}.json
    '''

    s = f'''#!/bin/bash
#SBATCH --job-name=classifier_artifacts
#SBATCH --mail-user=adam.amster@alleninstitute.org # Send mail to your AI email
#SBATCH --mail-type=BEGIN,END,FAIL # Send mail on begin, end, fail
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --output={log_path}/%A-%a.log
#SBATCH --partition braintv
#SBATCH --array=0-{len(exp_ids)-1}
#SBATCH --ntasks=24
    
readarray -t exp_ids < {base_path / 'exp_ids.txt'}
exp_id={exp_id}
    
{cmd}
    '''

    with open(slurm_scripts_dir /
              f'threshold_scaling_{threshold_scaling}.sh', 'w') as f:
        f.write(s)


def write_model_inputs_json(
        rois: List[Dict],
        exp_ids: List[str],
        threshold_scaling: int
):
    base_path = Path('/allen/aibs/informatics/aamster/segmentation/'
                     'threshold_scaling_analysis/classifier_inference/')
    model_inputs_dir = base_path / 'model_inputs' / \
        f'threshold_scaling_{threshold_scaling}'
    os.makedirs(model_inputs_dir, exist_ok=True)

    thumbnail_inputs_dir = Path(
        '/allen/aibs/informatics/aamster/segmentation/'
        'threshold_scaling_analysis/classifier_thumbnails/')
    thumbnail_inputs_path = \
        thumbnail_inputs_dir / f'threshold_scaling_{threshold_scaling}'

    for exp_id in exp_ids:
        exp_rois = [x for x in rois if x['experiment_id'] == exp_id]
        d = [{
            'experiment_id': exp_id,
            'roi_id': roi['id'],
            'mask_path': str(thumbnail_inputs_path /
                             f'mask_{exp_id}_{roi["id"]}.png'),
            'max_projection_path': str(thumbnail_inputs_path /
                                       f'max_{exp_id}_{roi["id"]}.png'),
            'correlation_projection_path': str(
                    thumbnail_inputs_path /
                    f'correlation_{exp_id}_{roi["id"]}.png')
        } for roi in exp_rois]
        with open(model_inputs_dir / f'{exp_id}_model_inputs.json', 'w') as f:
            f.write(json.dumps(d, indent=2))


def write_classifier_inference_jsons(
        exp_ids: List[str],
        threshold_scaling: int
):
    base_path = Path('/allen/aibs/informatics/aamster/segmentation/'
                     'threshold_scaling_analysis/classifier_inference/')
    preds_out_dir = base_path / f'threshold_scaling_{threshold_scaling}' / \
              'predictions'
    input_jsons_out_dir = base_path / \
                          f'threshold_scaling_{threshold_scaling}' / \
                          'input_jsons'

    model_inputs_dir = base_path / 'model_inputs' / \
        f'threshold_scaling_{threshold_scaling}'
    model_dir = base_path / 'model'

    os.makedirs(preds_out_dir, exist_ok=True)
    os.makedirs(input_jsons_out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    s3 = boto3.client('s3')
    sagemaker_train_job_ids = (
        'deepcell-train-fold-0-1653686076',
        'deepcell-train-fold-1-1653686083',
        'deepcell-train-fold-2-1653686088',
        'deepcell-train-fold-3-1653686094',
        'deepcell-train-fold-4-1653686099'
    )
    for i, train_job in enumerate(sagemaker_train_job_ids):
        if not (model_dir / f'{i}_model.pt').exists():
            with open(model_dir / f'model_{i}.tar.gz', 'wb') as f:
                s3.download_fileobj('dev.deepcell.alleninstitute.org',
                                    f'{train_job}/output/model.tar.gz', f)

            with tarfile.open(model_dir / f'model_{i}.tar.gz') as tar:
                tar.extractall(model_dir)
                shutil.move(str(model_dir / f'{i}' / f'{i}_model.pt'),
                            str(model_dir))
                os.remove(str(model_dir / f'model_{i}.tar.gz'))
                shutil.rmtree(str(model_dir / f'{i}'))

    for exp_id in exp_ids:
        d = {
            "model_params": {
                "use_pretrained_model": True,
                "model_architecture": "vgg11_bn",
                "truncate_to_layer": 22
            },
            "model_inputs_paths": [
                str(model_inputs_dir / f'{exp_id}_model_inputs.json')
            ],
            "model_load_path": str(model_dir),
            "save_path": str(preds_out_dir),
            "mode": "production",
            "experiment_id": exp_id
        }
        with open(input_jsons_out_dir / f'{exp_id}.json', 'w') as f:
            f.write(json.dumps(d, indent=2))


def write_slurm_script_for_model_inference(
        exp_ids: List[str],
        threshold_scaling: int
):
    base_path = Path('/allen/aibs/informatics/aamster/segmentation/'
                     'threshold_scaling_analysis/')
    classifier_inference_dir = Path('/allen/aibs/informatics/aamster/segmentation/'
                     'threshold_scaling_analysis/classifier_inference/')
    classifier_inference_dir_input_dir = classifier_inference_dir / \
                          f'threshold_scaling_{threshold_scaling}' / \
                          'input_jsons'
    out_dir = classifier_inference_dir / 'slurm'
    logs_dir = out_dir / 'logs'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    exp_id = '${exp_ids[$SLURM_ARRAY_TASK_ID]}'

    s = f'''#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --mail-user=adam.amster@alleninstitute.org # Send mail to your AI email
#SBATCH --mail-type=BEGIN,END,FAIL # Send mail on begin, end, fail
#SBATCH --mem=32gb
#SBATCH --time=48:00:00
#SBATCH --output={logs_dir}/%A-%a.log
#SBATCH --partition braintv
#SBATCH --ntasks=24
#SBATCH --array=0-{len(exp_ids)-1}

readarray -t exp_ids < {base_path / 'exp_ids.txt'}

/allen/aibs/informatics/aamster/miniconda3/envs/deepcell_new/bin/python -m \
    deepcell.cli.modules.inference \
    --input_json {classifier_inference_dir_input_dir / f'{exp_id}.json'} 
    '''

    with open(out_dir / f'threshold_scaling_{threshold_scaling}.sh', 'w') as f:
        f.write(s)


def _plot_groundtruth_and_pred_on_projection_for_exp_id(
        exp_id: str,
        groundtruth_rois: Dict,
        pred_rois: Dict,
        threshold_scaling: int,
        ax,
        plot_column,
        groundtruth_labels: Optional[pd.DataFrame] = None,
        preds: Optional[pd.DataFrame] = None,
        include_errors=False,
        roi_assignment=None
):
    plot_title = f'threshold-scaling={threshold_scaling}'

    if groundtruth_labels is not None:
        groundtruth_labels = groundtruth_labels.set_index(
            ['experiment_id', 'roi_id'])
        groundtruth_labels = groundtruth_labels.sort_index()
    if preds is not None:
        preds = preds.set_index(
            ['experiment_id', 'roi_id'])
        preds = preds.sort_index()
    with h5py.File('/allen/programs/mindscope/workgroups/surround/'
                   f'denoising_labeling_2022/denoised_movies/{exp_id}/'
                   f'{exp_id}_max.h5', 'r') as f:
        max_proj = f['max_projeciton'][:]

    low, high = np.quantile(max_proj, (0.2, 0.99))
    max_proj[max_proj <= low] = low
    max_proj[max_proj >= high] = high
    max_proj = (max_proj - max_proj.min()) / (max_proj.max() - max_proj.min())
    max_proj *= 255
    max_proj = max_proj.astype('uint8')
    max_proj = np.stack([max_proj, max_proj, max_proj], axis=-1)

    def plot_roi_contours_on_projection(roi, color):
        roi = extract_roi_to_ophys_roi(roi)
        pixel_array = roi.global_pixel_array.transpose()

        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[pixel_array[0], pixel_array[1]] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(max_proj, contours, -1, color, 1)

    if include_errors:
        false_negative_rois = {
            (roi_id, exp_id): x
            for ((roi_id, exp_id), x) in groundtruth_rois.items()
            if roi_id in roi_assignment['gt_unassigned']
        }

        tp_groundtruth_rois = {
            (roi_id, exp_id): x
            for ((roi_id, exp_id), x) in groundtruth_rois.items()
            if roi_id in roi_assignment['gt_assigned']
        }

        tp_pred_rois = {
            (roi_id, exp_id): x
            for ((roi_id, exp_id), x) in pred_rois.items()
            if roi_id in roi_assignment['pred_assigned']
        }

        false_positive_rois = {
            (roi_id, exp_id): x
            for ((roi_id, exp_id), x) in pred_rois.items()
            if roi_id in roi_assignment['pred_unassigned']
        }
        for (_, _), roi in tp_groundtruth_rois.items():
            plot_roi_contours_on_projection(roi=roi, color=(0, 255, 0))

        # for (_, _), roi in tp_pred_rois.items():
        #     plot_roi_contours_on_projection(roi=roi, color=(0, 0, 255))

        for (_, _), roi in false_negative_rois.items():
            plot_roi_contours_on_projection(roi=roi, color=(254, 221, 0))

        for (_, _), roi in false_positive_rois.items():
            plot_roi_contours_on_projection(roi=roi, color=(255, 0, 0))

        plot_title += f'\nTP={len(tp_groundtruth_rois)}  ' \
                      f'FP={len(false_positive_rois)}  ' \
                      f'FN={len(false_negative_rois)}'
    else:
        for (_, _), roi in groundtruth_rois.items():
            label = groundtruth_labels.loc[(exp_id, roi['id'])]['label']
            if label != 'cell':
                continue
            plot_roi_contours_on_projection(roi=roi, color=(0, 255, 0))

        for (_, _), roi in pred_rois.items():
            pred_label = preds.loc[(exp_id, roi['id'])].iloc[0]['y_pred']
            if pred_label != 'cell':
                continue
            plot_roi_contours_on_projection(roi=roi, color=(0, 0, 255))

    ax[plot_column].imshow(max_proj, cmap='gray')
    ax[plot_column].set_title(plot_title)


def plot_groundtruth_and_pred_on_projection(
        exp_ids: List[str],
        groundtruth_labels: pd.DataFrame,
        groundtruth_rois: Dict,
        labeled_regions: List[JobRegion]
):
    fig, ax = plt.subplots(nrows=1, ncols=3,
                           figsize=(3*int(512/100), int(512/100)), dpi=100)

    for exp_id in [exp_ids[0]]:
        exp_groundtruth_rois = \
            {(roi_id, exp_id): x
             for ((roi_id, exp_id), x) in groundtruth_rois.items()
             if x['experiment_id'] == exp_id}

        for i, threshold_scaling in enumerate(range(1, 4)):
            pred_rois = get_rois_in_labeled_regions(
                exp_ids=list(exp_ids),
                regions=labeled_regions,
                threshold_scaling=threshold_scaling)
            pred_rois = {
                (x['id'], x['experiment_id']): x
                for x in pred_rois
            }
            preds_out_dir = Path(
                '/allen/aibs/informatics/aamster/segmentation/'
                'threshold_scaling_analysis/classifier_inference/') / \
                f'threshold_scaling_{threshold_scaling}' / \
                'predictions' / f'{exp_id}_inference.csv'
            preds = pd.read_csv(preds_out_dir, dtype={'experiment_id': str})
            exp_pred_rois = \
                {(roi_id, exp_id): x
                 for ((roi_id, exp_id), x) in pred_rois.items()
                 if x['experiment_id'] == exp_id}

            _plot_groundtruth_and_pred_on_projection_for_exp_id(
                exp_id=exp_id,
                threshold_scaling=threshold_scaling,
                groundtruth_rois=exp_groundtruth_rois,
                groundtruth_labels=groundtruth_labels,
                pred_rois=exp_pred_rois,
                preds=preds,
                ax=ax,
                plot_column=i
            )
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='yellow', lw=4)]
    fig.legend(custom_lines, ['Ground truth', 'Pred'])
    plt.show()


def get_segmentation_performance_for_exp_id(
    exp_id: str,
    groundtruth_labels: pd.DataFrame,
    groundtruth_rois: Dict,
    labeled_regions: List[JobRegion]
):
    roi_assignment = dict()
    res = []

    gt_roi_ids = set([roi_id for roi_id, _ in groundtruth_rois])

    groundtruth_cells = \
        groundtruth_labels[
            (groundtruth_labels['label'] == 'cell') &
            (groundtruth_labels['experiment_id'] == exp_id) &
            (groundtruth_labels['roi_id'].isin(gt_roi_ids))]

    for i, threshold_scaling in enumerate(range(1, 4)):
        pred_rois = get_rois_in_labeled_regions_for_exp_id(
            exp_id=exp_id,
            regions=labeled_regions,
            threshold_scaling=threshold_scaling)
        pred_rois = {
            (x['id'], x['experiment_id']): x
            for x in pred_rois
        }

        pred_roi_ids = set([roi_id for roi_id, _ in pred_rois])

        preds_out_dir = Path(
            '/allen/aibs/informatics/aamster/segmentation/'
            'threshold_scaling_analysis/classifier_inference/') / \
                        f'threshold_scaling_{threshold_scaling}' / \
                        'predictions' / f'{exp_id}_inference.csv'
        preds = pd.read_csv(preds_out_dir, dtype={'experiment_id': str})
        preds = preds[~preds.duplicated(subset=['roi_id', 'experiment_id'])]

        pred_cells = preds[
            (preds['y_pred'] == 'cell') &
            (preds['experiment_id'] == exp_id) &
            (preds['roi_id'].isin(pred_roi_ids))
            ]

        gt_assigned = set()
        pred_assigned = set()
        unassigned_gt = set()
        unassigned_pred = set()

        for gt in groundtruth_cells.itertuples(index=False):
            gt_roi = groundtruth_rois[(gt.roi_id, exp_id)]

            for pred in pred_cells.itertuples(index=False):
                if pred.roi_id in pred_assigned:
                    continue
                pred_roi = pred_rois[(pred.roi_id, exp_id)]

                iou = calc_iou(roi0=gt_roi, roi1=pred_roi)
                if iou > 0.65:
                    gt_assigned.add(gt.roi_id)
                    pred_assigned.add(pred.roi_id)
                    break
        for gt in groundtruth_cells.itertuples(index=False):
            if gt.roi_id not in gt_assigned:
                unassigned_gt.add(gt.roi_id)

        for pred in pred_cells.itertuples(index=False):
            if pred.roi_id not in pred_assigned:
                unassigned_pred.add(pred.roi_id)

        roi_assignment[threshold_scaling] = {
            'gt_assigned': gt_assigned,
            'pred_assigned': pred_assigned,
            'gt_unassigned': unassigned_gt,
            'pred_unassigned': unassigned_pred
        }
        n = groundtruth_cells.shape[0]
        tp = len(gt_assigned)
        fp = len(unassigned_pred)
        fn = len(unassigned_gt)

        assert len(gt_assigned) + len(unassigned_gt) == \
               groundtruth_cells.shape[0]
        assert len(pred_assigned) + len(unassigned_pred) == \
                pred_cells.shape[0]

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = np.nan
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = np.nan

        try:
            f1 = tp / (tp + ((fp + fn) / 2))
        except ZeroDivisionError:
            f1 = np.nan
        res.append({
            'experiment_id': exp_id,
            'threshold_scaling': threshold_scaling,
            'n': n,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    res = pd.DataFrame(res)
    return res, roi_assignment


def plot_segmentation_performance(
        exp_id: str,
        roi_assignment: Dict,
        groundtruth_rois: Dict,
        labeled_regions: List[JobRegion]
):
    fig, ax = plt.subplots(nrows=1, ncols=3,
                           figsize=(3*int(512/100), int(512/100)), dpi=100)

    for i, threshold_scaling in enumerate(range(1, 4)):
        pred_rois = get_rois_in_labeled_regions_for_exp_id(
            exp_id=exp_id,
            regions=labeled_regions,
            threshold_scaling=threshold_scaling)
        pred_rois = {
            (x['id'], x['experiment_id']): x
            for x in pred_rois
        }
        groundtruth_rois = {
            (roi_id, exp_id_): v
            for (roi_id, exp_id_), v in groundtruth_rois.items()
            if exp_id_ == exp_id
        }
        _plot_groundtruth_and_pred_on_projection_for_exp_id(
            exp_id=exp_id,
            groundtruth_rois=groundtruth_rois,
            pred_rois=pred_rois,
            ax=ax,
            plot_column=i,
            roi_assignment=roi_assignment[threshold_scaling],
            threshold_scaling=threshold_scaling,
            include_errors=True
        )

    custom_lines = [Line2D([0], [0], color='green', lw=4),
                    # Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color=(254/255, 221/255, 0), lw=4)]
    fig.legend(custom_lines, ['TP', 'FP', 'FN'])
    plt.show()


def get_segmentation_performance(
        exp_ids: List[str],
        groundtruth_labels: pd.DataFrame,
        groundtruth_rois: Dict,
        labeled_regions: List[JobRegion]
):
    res = []
    for exp_id in tqdm(exp_ids):
        res_, _ = get_segmentation_performance_for_exp_id(
            exp_id=exp_id,
            groundtruth_labels=groundtruth_labels,
            groundtruth_rois=groundtruth_rois,
            labeled_regions=labeled_regions
        )
        res.append(res_)
    res = pd.concat(res)
    return res


def calc_iou(roi0, roi1):
    roi0 = extract_roi_to_ophys_roi(roi0)
    roi0_pixel_array = roi0.global_pixel_array.transpose()

    roi0_mask = np.zeros((512, 512), dtype=np.uint8)
    roi0_mask[roi0_pixel_array[0], roi0_pixel_array[1]] = 1

    roi1 = extract_roi_to_ophys_roi(roi1)
    roi1_pixel_array = roi1.global_pixel_array.transpose()

    roi1_mask = np.zeros((512, 512), dtype=np.uint8)
    roi1_mask[roi1_pixel_array[0], roi1_pixel_array[1]] = 1

    intersection = (roi0_mask * roi1_mask).sum()
    union = (roi0_mask.sum() +
             roi1_mask.sum() -
             intersection)

    return intersection / union


def plot_threshold_scaling_performance(

):
    perf = pd.read_csv('/tmp/threshold_scaling_analysis.csv',
                       dtype={'experiment_id': str})
    experiment_meta = _get_experiment_meta_dataframe()
    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=experiment_meta)
    experiment_meta['experiment_id'] = \
        experiment_meta['experiment_id'].astype(str)
    experiment_meta = experiment_meta.set_index('experiment_id')
    perf['depth_bin'] = perf['experiment_id'].map(
        experiment_meta['depth_bin'])
    perf = perf[~perf.duplicated(subset=['experiment_id', 'f1'])]

    sorted_depth_bins = np.array(_get_sorted_depth_bin_labels(
        experiment_meta=experiment_meta
    ))
    perf['depth_bin_sorted'] = perf['depth_bin'].apply(
        lambda x: np.argwhere(sorted_depth_bins == x)[0][0])
    perf = perf.sort_values('depth_bin_sorted')
    for metric in ('f1', 'precision', 'recall'):
        g = sns.barplot(data=perf, x='depth_bin', y=metric,
                    hue='threshold_scaling'
                    )
        plt.title(f'Mean {metric} by depth bin')
        yield g


def _get_sorted_depth_bin_labels(experiment_meta: pd.DataFrame):
    depth_bins = experiment_meta['depth_bin'].unique().tolist()
    depth_bins = sorted(depth_bins,
                        key=lambda x: int(re.findall(r'\[(\d+)', x)[0]))
    return depth_bins


def _get_experiment_meta_dataframe(
        experiment_metadata_path='/allen/aibs/informatics/chris.morrison/'
                                 'ticket-29/experiment_metadata.json'):
    experiment_meta = pd.read_json(experiment_metadata_path)
    experiment_meta = experiment_meta.T

    with open(experiment_metadata_path) as f:
        experiment_ids = json.load(f).keys()
    experiment_meta['experiment_id'] = experiment_ids
    experiment_meta['experiment_id'] = experiment_meta['experiment_id'] \
        .astype(int)
    experiment_meta = experiment_meta.reset_index(drop=True)
    return experiment_meta


def _get_experiment_meta_with_depth_bin(experiment_meta: pd.DataFrame):
    exp_metas = []
    for row in experiment_meta.itertuples(index=False):
        exp_metas.append(ExperimentMetadata(
            experiment_id=row.experiment_id,
            imaging_depth=row.imaging_depth,
            equipment=row.equipment,
            problem_experiment=False
        ))

    exp_bin_ids, bin_edges = \
        DataSplitter._get_experiment_groups_for_stratified_split(
            exp_metas, n_depth_bins=5)

    bin_edge_str = [
                       f'[0, {bin_edges[0]:.0f})'
                   ] + [f'[{bin_edges[i - 1]:.0f}, {bin_edges[i]:.0f})' for i
                        in
                        range(1, len(bin_edges))]

    bin_edge_str_map = {i: bin_edge_str[i] for i in range(len(bin_edges))}
    experiment_meta['bin_id'] = exp_bin_ids
    experiment_meta['depth_bin'] = experiment_meta['bin_id'].map(
        bin_edge_str_map)
    return experiment_meta


def generate_pdf_of_results(
        exp_ids: List[str],
        groundtruth_rois: Dict,
        regions: List[JobRegion],
        groundtruth_labels: pd.DataFrame
):
    with PdfPages('/local1/threshold_scaling_analysis.pdf') as pdf:
        for _ in plot_threshold_scaling_performance():
            pdf.savefig()
            plt.close()

        experiment_meta = _get_experiment_meta_dataframe()
        experiment_meta = _get_experiment_meta_with_depth_bin(
            experiment_meta=experiment_meta)
        experiment_meta['experiment_id'] = \
            experiment_meta['experiment_id'].astype(str)
        experiment_meta = experiment_meta.set_index('experiment_id')
        experiment_meta = experiment_meta.loc[exp_ids]
        sorted_depth_bins = np.array(_get_sorted_depth_bin_labels(
            experiment_meta=experiment_meta
        ))
        experiment_meta['depth_bin_sorted'] = experiment_meta['depth_bin'].apply(
            lambda x: np.argwhere(sorted_depth_bins == x)[0][0])
        experiment_meta = experiment_meta.sort_values(by='depth_bin_sorted')

        for exp_id in tqdm(experiment_meta.index):
            perf, roi_assignment = get_segmentation_performance_for_exp_id(
                exp_id=exp_id,
                groundtruth_rois=groundtruth_rois,
                groundtruth_labels=groundtruth_labels,
                labeled_regions=regions
            )
            perf = perf.set_index('experiment_id')

            plot_segmentation_performance(
                exp_id=exp_id,
                groundtruth_rois=groundtruth_rois,
                labeled_regions=regions,
                roi_assignment=roi_assignment
            )
            exp_regions = [x for x in regions
                           if str(x.experiment_id) == exp_id]
            fov_labeled = len(exp_regions) / 9
            n_cells = perf.loc[exp_id]['n'].iloc[0]

            perf['depth_bin'] = perf.index.map(
                experiment_meta['depth_bin'])

            depth_bin = perf.loc[exp_id]['depth_bin'].iloc[0]

            plt.suptitle(f'Experiment id {exp_id}\nN cells: {n_cells}, '
                         f'FOV frac: {fov_labeled:.3f}, '
                         f'depth bin: {depth_bin}\n')
            pdf.savefig()
            plt.close()


def main():
    test_experiments = get_test_experiments()

    labeling_app_config, labeling_app_context = _get_labeling_app()
    with labeling_app_context:
        regions = get_labeled_regions_for_experiments(exp_ids=test_experiments)
        groundtruth_labels = construct_dataset(
            label_db_path=(labeling_app_config['SQLALCHEMY_DATABASE_URI']
                .replace('sqlite:///', '')),
            min_labelers_per_roi=(
                labeling_app_config['LABELERS_REQUIRED_PER_REGION']),
            vote_tallying_strategy=VoteTallyingStrategy.MAJORITY
        )
        groundtruth_labels['roi_id'] = groundtruth_labels['roi_id'].astype(int)
        groundtruth_labels = groundtruth_labels[
            groundtruth_labels['experiment_id'].isin(test_experiments)]
        groundtruth_rois = get_rois_in_labeled_regions(
            exp_ids=list(test_experiments),
            regions=regions,
            ground_truth=True,
            groundtruth_labels=groundtruth_labels
        )
        assert len(groundtruth_rois) == \
               groundtruth_labels[groundtruth_labels['label'] == 'cell'].shape[0]
        groundtruth_rois = {
            (x['id'], x['experiment_id']): x
            for x in groundtruth_rois
        }

    # for threshold_scaling in range(1, 4):
        # rois = get_rois_in_labeled_regions(
        #     exp_ids=list(test_experiments),
        #     regions=regions,
        #     threshold_scaling=threshold_scaling)
        # write_slurm_script_for_artifact_creation(
        #     exp_ids=list(test_experiments),
        #     threshold_scaling=threshold_scaling,
        #     rois=rois)
        # write_model_inputs_json(
        #     rois=rois,
        #     exp_ids=list(test_experiments),
        #     threshold_scaling=threshold_scaling
        # )
        # write_classifier_inference_jsons(
        #     exp_ids=list(test_experiments),
        #     threshold_scaling=threshold_scaling
        # )
        # write_slurm_script_for_model_inference(
        #     exp_ids=list(test_experiments),
        #     threshold_scaling=threshold_scaling
        # )
    # plot_groundtruth_and_pred_on_projection(
    #     groundtruth_labels=groundtruth_labels,
    #     groundtruth_rois=groundtruth_rois,
    #     labeled_regions=regions,
    #     exp_ids=list(test_experiments)
    # )

    # perf = get_segmentation_performance(
    #     exp_ids=list(test_experiments),
    #     groundtruth_labels=groundtruth_labels,
    #     groundtruth_rois=groundtruth_rois,
    #     labeled_regions=regions
    # )
    # perf.to_csv('/tmp/threshold_scaling_analysis.csv', index=False)
    # perf, roi_assignment = get_segmentation_performance_for_exp_id(
    #     exp_id='793618875',
    #     groundtruth_rois=groundtruth_rois,
    #     groundtruth_labels=groundtruth_labels,
    #     labeled_regions=regions
    # )
    # plot_segmentation_performance(
    #     exp_id='793618875',
    #     groundtruth_rois=groundtruth_rois,
    #     labeled_regions=regions,
    #     roi_assignment=roi_assignment
    # )
    generate_pdf_of_results(
        exp_ids=list(test_experiments),
        groundtruth_rois=groundtruth_rois,
        groundtruth_labels=groundtruth_labels,
        regions=regions
    )


if __name__ == '__main__':
    main()
