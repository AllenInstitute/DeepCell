import argparse
import json
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
import seaborn as sns

import pandas as pd
from PIL import Image
from deepcell.cli.modules.create_dataset import construct_dataset
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from ophys_etl.utils.rois import sanitize_extract_roi_list, \
    extract_roi_to_ophys_roi
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm

from scripts.data_exploration import _get_roi_label_agreement, _get_experiment_meta_with_depth_bin, \
    _get_experiment_meta_dataframe, _get_sorted_depth_bin_labels, \
    _get_classifier_scores, _get_confusion_matrix_by_depth
from src.deepcell.plotting import plot_confusion_matrix, plot_pr_curve


def get_disagreement_by_depth(labels: pd.DataFrame, targets: pd.DataFrame):
    agreement = _get_roi_label_agreement(
        labels=labels,
        targets=targets
    )
    agreement['experiment_id'] = agreement['experiment_id'].astype(str)

    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=_get_experiment_meta_dataframe()
    )
    experiment_meta['experiment_id'] = experiment_meta['experiment_id'].astype(str)

    agreement = agreement.merge(experiment_meta, on='experiment_id')

    agreement_by_depth = \
        agreement.groupby(['depth_bin', 'label_target'])['is_consensus'].mean()
    agreement_by_depth = agreement_by_depth.reset_index()
    agreement_by_depth = \
        agreement_by_depth.rename(
            columns={'is_consensus': 'consensus_rate', 'label_target': 'label'})
    sns.barplot(data=agreement_by_depth,
                x='depth_bin', y='consensus_rate',
                hue='label',
                hue_order=('not cell', 'cell'),
                order=_get_sorted_depth_bin_labels(
                        experiment_meta=experiment_meta))
    plt.show()


def get_perf_by_fov(preds_path: str):
    scores = _get_classifier_scores(preds_path=Path(preds_path))
    scores['y_pred'] = scores['y_pred'].apply(lambda x: 'cell' if x else 'not cell')

    def get_perf(x):
        precision = precision_score(y_true=x['y_true'], y_pred=x['y_pred'],
                                    pos_label='cell', zero_division=1)
        recall = recall_score(y_true=x['y_true'], y_pred=x['y_pred'],
                                    pos_label='cell', zero_division=1)
        tp = ((x['y_pred'] == 'cell') & (x['y_true'] == 'cell')).sum()
        fp = ((x['y_pred'] == 'cell') & (x['y_true'] == 'not cell')).sum()
        fn = ((x['y_pred'] == 'not cell') & (x['y_true'] == 'cell')).sum()

        return pd.DataFrame({'metric': ['precision', 'recall'],
                             'value': [precision, recall],
                             'denom': [tp+fp, tp+fn]})
    perf = scores.groupby(['experiment_id']).apply(get_perf)
    perf = perf.reset_index()

    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=_get_experiment_meta_dataframe()
    )
    experiment_meta['experiment_id'] = \
        experiment_meta['experiment_id'].astype(str)

    perf = perf.merge(experiment_meta, on='experiment_id')

    zero_index_exp_id = perf.groupby('depth_bin')['experiment_id']\
        .apply(lambda x: x.map(
        {exp_id: i for (exp_id, i) in zip(x.unique(),
                                          range(len(x.unique())))
         }))
    perf['experiment_id'] = zero_index_exp_id

    # sort by depth bin
    sorted_depth_bins = _get_sorted_depth_bin_labels(experiment_meta=experiment_meta)
    perf = pd.concat([
        perf[perf['depth_bin'] == depth_bin]
        for depth_bin in sorted_depth_bins])

    # Hiding cases where there is too little data (and the model made some mistakes)
    # to make visualization easier to read
    perf = perf[(perf['denom'] > 2) |
                ((perf['denom'] <= 2) & (perf['value'] == 1.0))]

    g = sns.boxplot(data=perf, x='depth_bin',
                    y='value',
                    hue='metric')
    plt.suptitle('Test set performance across all FOV')

    plt.show()


def get_preds_by_fov_pdf(rois_path, preds_path: str, projections_path: str):
    scores = _get_classifier_scores(preds_path=Path(preds_path))
    scores['y_pred'] = scores['y_pred'].apply(lambda x: 'cell' if x else 'not cell')
    scores['y_true'] = scores['y_true'].apply(lambda x: 'cell' if x else 'not cell')

    scores['experiment_id'] = scores['experiment_id'].astype(str)

    exp_ids = scores['experiment_id'].unique()
    rois_path = Path(rois_path)
    projections_path = Path(projections_path)

    valid_set_rois = set(list(zip(scores['experiment_id'], scores['roi_id'])))

    scores = scores.set_index(['experiment_id', 'roi_id'])

    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=_get_experiment_meta_dataframe()
    )
    experiment_meta['experiment_id'] = \
        experiment_meta['experiment_id'].astype(str)

    depth_bin_exp_ids_map = \
        experiment_meta.groupby('depth_bin')['experiment_id'].unique()

    with PdfPages('/home/adam.amster/validation_results.pdf') as pdf:
        for depth_bin in tqdm(_get_sorted_depth_bin_labels(
                        experiment_meta=experiment_meta)):
            for exp_id in depth_bin_exp_ids_map[depth_bin]:
                if exp_id not in exp_ids:
                    continue
                with open(rois_path / exp_id / f'{exp_id}_rois.json') as f:
                    rois = json.load(f)
                with h5py.File(projections_path / f'{exp_id}_artifacts.h5') as f:
                    max_proj = f['max_projection'][()]

                # normalize contrast in projection
                low, high = np.quantile(max_proj, (0.2, 0.99))
                max_proj[max_proj <= low] = low
                max_proj[max_proj >= high] = high

                max_proj = (max_proj - max_proj.min()) / (max_proj.max() - max_proj.min())
                max_proj *= 255
                max_proj = max_proj.astype('uint8')
                max_proj = np.stack([max_proj, max_proj, max_proj], axis=-1)

                rois = [x for x in rois if (exp_id, x['id']) in valid_set_rois]
                rois = sanitize_extract_roi_list(rois)
                for roi in rois:
                    roi = extract_roi_to_ophys_roi(roi)
                    pixel_array = roi.global_pixel_array.transpose()

                    mask = np.zeros((512, 512), dtype=np.uint8)
                    mask[pixel_array[0], pixel_array[1]] = 255

                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)
                    pred = scores.loc[(exp_id, roi.roi_id)]['y_pred']
                    true = scores.loc[(exp_id, roi.roi_id)]['y_true']

                    if pred == 'cell' and true == 'cell':
                        color = (0, 255, 0)
                    elif pred == 'cell' and true == 'not cell':
                        color = (255, 0, 0)
                    elif pred == 'not cell' and true == 'cell':
                        color = (0, 0, 255)
                    else:
                        continue
                    cv2.drawContours(max_proj, contours, -1, color, 1)

                precision = precision_score(
                    y_true=scores.loc[exp_id]['y_true'],
                    y_pred=scores.loc[exp_id]['y_pred'],
                    pos_label='cell',
                    zero_division=1
                )

                if (scores.loc[exp_id]['y_true'] == 'cell').sum() == 0:
                    recall = np.nan
                else:
                    recall = recall_score(
                        y_true=scores.loc[exp_id]['y_true'],
                        y_pred=scores.loc[exp_id]['y_pred'],
                        pos_label='cell',
                        zero_division=1
                    )
                fig, ax = plt.subplots()

                plt.imshow(max_proj)
                custom_lines = [Line2D([0], [0], color='green', lw=4),
                                Line2D([0], [0], color='red', lw=4),
                                Line2D([0], [0], color='blue', lw=4)]
                ax.legend(custom_lines, ['TP', 'FP', 'FN'],
                          bbox_to_anchor=(1.04, 0.5),
                          loc="center left", borderaxespad=0)
                plt.title(f'Experiment {exp_id}\nDepth bin = {depth_bin}\n'
                          f'Precision: {precision:.2f}, Recall: {recall:.2f}',
                          fontdict={'fontsize': 10})
                pdf.savefig()
                plt.close()


def get_mistakes(preds_path: str, classifier_inputs_path: str, correlation_path: str):
    classifier_inputs_path = Path(classifier_inputs_path)
    correlation_path = Path(correlation_path)

    scores = _get_classifier_scores(preds_path=Path(preds_path))
    scores['y_pred'] = scores['y_pred'].apply(lambda x: 'cell' if x else 'not cell')
    scores['y_true'] = scores['y_true'].apply(lambda x: 'cell' if x else 'not cell')

    scores['experiment_id'] = scores['experiment_id'].astype(str)

    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=_get_experiment_meta_dataframe()
    )
    experiment_meta['experiment_id'] = \
        experiment_meta['experiment_id'].astype(str)

    scores = scores.merge(experiment_meta, on='experiment_id')

    scores = scores.set_index(['experiment_id', 'roi_id'])

    fp = scores[(scores['y_pred'] == 'cell') & (scores['y_true'] == 'not cell')]
    fn = scores[(scores['y_pred'] == 'not cell') & (scores['y_true'] == 'cell')]

    def get_classifier_score_color(classifier_score: float,
                                   color_map='viridis'):
        cmap = matplotlib.cm.get_cmap(color_map)

        color = tuple([int(255 * x) for x in cmap(classifier_score)][:-1])
        color = (color[0], color[1], color[2])
        return color

    with PdfPages('/home/adam.amster/mistakes.pdf') as pdf:
        for i, rows in enumerate((fp, fn)):
            for depth_bin in tqdm(_get_sorted_depth_bin_labels(
                            experiment_meta=experiment_meta)):
                rows_in_depth_bin = rows[rows['depth_bin'] == depth_bin]
                sample_size = min(rows_in_depth_bin.shape[0], 10)
                rows_in_depth_bin = rows_in_depth_bin.sample(sample_size)
                for row in rows_in_depth_bin.itertuples():
                    exp_id, roi_id = row.Index
                    with open(classifier_inputs_path / f'max_{exp_id}_{roi_id}.png', 'rb') as f:
                        max = Image.open(f)
                        max = np.array(max)
                    with open(correlation_path / f'correlation_{exp_id}_{roi_id}.png', 'rb') as f:
                        corr = Image.open(f)
                        corr = np.array(corr)
                    with open(classifier_inputs_path / f'max_activation_{exp_id}_{roi_id}.png', 'rb') as f:
                        max_activation = Image.open(f)
                        max_activation = np.array(max_activation)
                    with open(classifier_inputs_path / f'mask_{exp_id}_{roi_id}.png', 'rb') as f:
                        mask = Image.open(f)
                        mask = np.array(mask)

                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_NONE)

                    max = np.stack([max, max, max], axis=-1)
                    corr = np.stack([corr, corr, corr], axis=-1)
                    max_activation = np.stack([max_activation, max_activation, max_activation], axis=-1)

                    color = get_classifier_score_color(classifier_score=row.y_score)
                    cv2.drawContours(max, contours, -1, color, 1)
                    cv2.drawContours(corr, contours, -1, color, 1)
                    cv2.drawContours(max_activation, contours, -1, color, 1)

                    fig, ax = plt.subplots(nrows=1, ncols=3)
                    ax[0].imshow(max)
                    ax[0].set_title('Max projection', fontdict={'fontsize': 10})
                    ax[1].imshow(corr)
                    ax[1].set_title('Correlation projection', fontdict={'fontsize': 10})
                    ax[2].imshow(max_activation)
                    ax[2].set_title('Max activation', fontdict={'fontsize': 10})

                    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

                    # creating ScalarMappable
                    cmap = plt.get_cmap('viridis')
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    plt.colorbar(sm, fraction=0.046, pad=0.04)

                    subtitle = 'False positive' if i == 0 else 'False negative'
                    plt.suptitle(
                        f'Experiment {exp_id}, ROI {roi_id}\nDepth bin = {depth_bin}\n{subtitle}')
                    pdf.savefig()
                    plt.close()


def get_human_performance(labels: pd.DataFrame, targets: pd.DataFrame):
    agreement = _get_roi_label_agreement(labels=labels, targets=targets)
    agreement['experiment_id'] = agreement['experiment_id'].astype(str)
    agreement['roi_id'] = agreement['roi_id'].astype(int)
    agreement = agreement[['experiment_id', 'roi_id', 'is_consensus', 'fraction', 'label_target']]

    def get_labeler_label(x):
        if x['label_target'] == 'cell':
            if x['is_consensus']:
                return 'cell'
            else:
                return 'not cell'
        else:
            if x['is_consensus']:
                return 'not cell'
            else:
                return 'cell'

    agreement['label_labeler'] = agreement.apply(get_labeler_label, axis=1)

    plot_confusion_matrix(classifier_scores=agreement,
                          label_col='label_target',
                          pred_col='label_labeler')
    _get_confusion_matrix_by_depth(classifier_scores=agreement,
                                   label_col='label_target',
                                   pred_col='label_labeler')


def main():
    labels = construct_dataset(
        label_db_path=args.labels_db_path,
        min_labelers_per_roi=MIN_LABELERS_PER_ROI,
        vote_threshold=args.vote_threshold,
        raw=True
    )
    targets = construct_dataset(
        label_db_path=args.labels_db_path,
        min_labelers_per_roi=MIN_LABELERS_PER_ROI,
        vote_threshold=args.vote_threshold,
        raw=False
    )
    # get_classifier_performance(
    #     preds_path=args.test_preds_path,
    #     labels=labels,
    #     targets=targets
    # )
    # get_classifier_performance(
    #     preds_path=args.test_preds_path,
    #     labels=labels,
    #     targets=targets,
    #     by_depth=True
    # )
    # get_disagreement_by_depth(labels=labels, targets=targets)
    # get_perf_by_fov(preds_path=args.preds_path)
    # get_preds_by_fov_pdf(rois_path=args.rois_path,
    #                      preds_path=args.val_preds_path,
    #                      projections_path=args.projections_path)
    # get_mistakes(preds_path=args.val_preds_path,
    #              classifier_inputs_path=args.classifier_inputs_path,
    #              correlation_path=args.correlation_path)
    # get_human_performance(labels=labels, targets=targets)

    scores = _get_classifier_scores(preds_path=Path(args.val_preds_path))
    plot_pr_curve(scores=scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_preds_path', required=True)
    parser.add_argument('--val_preds_path', required=True)
    parser.add_argument('--rois_path', required=True)
    parser.add_argument('--projections_path', required=True)
    parser.add_argument('--classifier_inputs_path', required=True)
    parser.add_argument('--correlation_path', required=True)
    parser.add_argument('--labels_db_path', required=True)
    parser.add_argument('--vote_threshold', default=0.5, type=float)
    args = parser.parse_args()
    MIN_LABELERS_PER_ROI = 3
    main()
