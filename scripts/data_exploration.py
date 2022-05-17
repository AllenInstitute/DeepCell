import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import precision_score, recall_score, \
    ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit
from wordcloud import WordCloud

from deepcell.data_splitter import DataSplitter
from deepcell.datasets.exp_metadata import ExperimentMetadata

plt.style.use('ggplot')


def _get_train_test_experiments(experiment_metadata: pd.DataFrame):
    exp_metas = []
    for row in experiment_metadata.itertuples(index=False):
        exp_metas.append(ExperimentMetadata(
            experiment_id=row.experiment_id,
            imaging_depth=row.imaging_depth,
            equipment=row.equipment,
            problem_experiment=row.problem_experiment
        ))
    exp_ids = np.array([int(exp_meta['experiment_id'])
                        for exp_meta in exp_metas], dtype='uint64')
    exp_bin_ids, _ = DataSplitter._get_experiment_groups_for_stratified_split(
        exp_metas, n_depth_bins=5)
    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=0.3,
                                 random_state=1234)
    train_index, test_index = next(sss.split(exp_bin_ids,
                                             exp_bin_ids))
    return exp_ids[train_index], exp_ids[test_index]


def _plot_depth_distribution(experiment_meta: pd.DataFrame,
                             experiment_num_labels: pd.Series):
    experiment_num_labels = experiment_num_labels.reset_index().rename(
        columns={0: 'num_regions_labeled'})
    experiment_meta = experiment_meta.merge(experiment_num_labels,
                                            on='experiment_id')
    for num_regions_labeled in (1, 2, 3):
        experiment_meta[
            experiment_meta['num_regions_labeled'] == num_regions_labeled] \
            ['imaging_depth'].plot.hist(
            label=f'{num_regions_labeled}/9 of FOV labeled')
    plt.legend()
    plt.xlabel('Depth')
    plt.title('Depth distribution for thrice labeled regions')
    plt.show()


def _plot_amount_of_fov_labeled(experiment_num_labels: pd.Series,
                                experiment_meta: pd.DataFrame,
                                job_regions: pd.DataFrame,
                                by_train_test_split=False):
    # add experiments with 0 labels
    experiment_num_labels = pd.concat([
        experiment_num_labels,
        pd.Series(0, index=pd.Int64Index(set(job_regions['experiment_id'])
                                         .difference(
            experiment_num_labels.index), name='experiment_id'))])

    experiment_frac_fov_labeled = experiment_num_labels.apply(lambda x: x / 9) \
        .reset_index().rename(columns={0: 'amount_of_FOV_labeled'})
    experiment_meta = experiment_meta.merge(experiment_frac_fov_labeled,
                                            on='experiment_id')
    experiment_meta = experiment_meta.sort_values('bin_id')

    train_exp, test_exp = _get_train_test_experiments(
        experiment_metadata=experiment_meta)

    experiment_meta = experiment_meta.set_index('experiment_id')
    experiment_meta.loc[train_exp, 'dataset'] = 'train'
    experiment_meta.loc[test_exp, 'dataset'] = 'test'
    experiment_meta = experiment_meta.reset_index()

    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=experiment_meta)

    experiment_meta['amount_of_FOV_labeled'] = \
        experiment_meta['amount_of_FOV_labeled'].apply(
            lambda x: f'{x:.1f}').astype('category')

    row = 'dataset' if by_train_test_split else None
    g = sns.FacetGrid(experiment_meta, col='depth_bin', row=row,
                      col_order=_get_sorted_depth_bin_labels(
                          experiment_meta=experiment_meta))
    g.map_dataframe(sns.histplot, x="amount_of_FOV_labeled")

    g = _add_n_samples_to_title(fig=g, experiment_meta=experiment_meta,
                                by_train_test_split=by_train_test_split)
    g.fig.subplots_adjust(top=0.8)
    plt.show()

    sns.histplot(experiment_meta, x='amount_of_FOV_labeled')
    plt.title('Amount of FOV labeled')
    plt.show()


def _plot_train_test_distribution(job_regions: pd.DataFrame,
                                  completed_labels: pd.DataFrame,
                                  experiment_meta: pd.DataFrame):
    train_exp, test_exp = _get_train_test_experiments(
        experiment_metadata=(
            experiment_meta[experiment_meta['experiment_id']
                .isin(completed_labels['experiment_id'])]))

    job_regions['imaging_depth'] \
        .plot.hist(label='all')
    completed_labels \
        .drop_duplicates(subset='region_id')['imaging_depth'] \
        .plot.hist(label='thrice labeled')
    completed_labels[completed_labels['experiment_id'].isin(train_exp)] \
        .drop_duplicates(subset='region_id')['imaging_depth'] \
        .plot.hist(label='train')
    completed_labels[completed_labels['experiment_id'].isin(test_exp)] \
        .drop_duplicates(subset='region_id')['imaging_depth'] \
        .plot.hist(label='test')
    plt.legend()
    plt.xlabel('Depth')
    plt.show()


def plot_duration_trends():
    df = pd.read_csv('~/Downloads/user_labels.csv', parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    df = df.groupby('user_id').filter(lambda x: x.shape[0] > 30)
    iqr = df['duration'].quantile(.75) - df['duration'].quantile(.25)
    df = df[df['duration'] < df['duration'].quantile(.75) + 1.5 * iqr]
    df['timestamp'] = df.groupby('user_id')['timestamp'].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))
    df['duration'] = df.groupby('user_id')['duration'].apply(
        lambda x: (x - x.min()) / (x.max() - x.min()))

    g = sns.FacetGrid(df, row='user_id')
    g.map(sns.scatterplot, 'timestamp', 'duration')

    print(df.groupby('user_id').apply(
        lambda x: pearsonr(x['timestamp'], x['duration'])))
    plt.show()


def plot_distribution_of_depths(user_labels: pd.DataFrame,
                                job_regions: pd.DataFrame):
    user_labels = user_labels.merge(job_regions, left_on='region_id',
                                    right_on='id')

    experiment_meta = _get_experiment_meta_dataframe()

    user_labels = user_labels.merge(experiment_meta, on='experiment_id')
    job_regions = job_regions.merge(experiment_meta, on='experiment_id')
    region_num_labels = user_labels.groupby('region_id').size()

    completed_regions = region_num_labels[region_num_labels >= 3].index
    completed_labels = user_labels[user_labels['region_id']
        .isin(completed_regions)]

    experiment_num_labels = completed_labels.drop_duplicates(
        subset='region_id').groupby('experiment_id').size()

    _plot_train_test_distribution(
        job_regions=job_regions,
        completed_labels=completed_labels,
        experiment_meta=experiment_meta
    )

    _plot_depth_distribution(
        experiment_meta=experiment_meta,
        experiment_num_labels=experiment_num_labels
    )

    _plot_amount_of_fov_labeled(
        experiment_num_labels=experiment_num_labels,
        experiment_meta=experiment_meta,
        job_regions=job_regions
    )

    _plot_amount_of_fov_labeled(
        experiment_num_labels=experiment_num_labels,
        experiment_meta=experiment_meta,
        job_regions=job_regions,
        by_train_test_split=True
    )


def get_overall_target_distribution(
        user_labels: pd.DataFrame,
        job_regions: pd.DataFrame
):
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions)
    targets = _get_targets(labels=labels,
                           filter_by_maximally_labeled_fov=False)

    target_counts = targets.groupby('label').size()
    target_counts = target_counts.reset_index()
    target_counts = target_counts.rename(columns={0: 'count'})
    target_counts['frac'] = target_counts['count'] / target_counts[
        'count'].sum()
    print(target_counts)


def get_cell_count_distribution(
        user_labels: pd.DataFrame,
        job_regions: pd.DataFrame,
        by_depth_bin=False,
        cell_probability=False
):
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions)
    targets = _get_targets(labels=labels,
                           filter_by_maximally_labeled_fov=False)

    cells = targets[targets['label'] == 'cell']
    n_cells_per_fov = cells.groupby('experiment_id').size()

    # Add regions with 0 cells
    n_cells_per_fov = pd.concat([
        n_cells_per_fov,
        pd.Series(0,
                  index=pd.Int64Index(
                      set(labels['experiment_id'].unique())
                          .difference(n_cells_per_fov.index),
                      name='experiment_id'))])

    if cell_probability:
        non_cells = targets[targets['label'] == 'not cell']
        n_noncells_per_fov = non_cells.groupby('experiment_id').size()

        # Add regions with 0 noncells
        n_noncells_per_fov = pd.concat([
            n_noncells_per_fov,
            pd.Series(0,
                      index=pd.Int64Index(
                          set(labels['experiment_id'].unique())
                              .difference(n_noncells_per_fov.index),
                          name='experiment_id'))])
        cell_stats = n_cells_per_fov / (n_cells_per_fov +
                                           n_noncells_per_fov)
    else:
        cell_stats = n_cells_per_fov

    cell_stats = cell_stats.reset_index()
    stat_name = 'Cell fraction' if cell_probability else \
        'Number of cells'
    cell_stats = cell_stats.rename(
        columns={0: stat_name})

    title = 'Cell fraction by depth across FOV' if cell_probability else \
        'Number of cells per FOV'

    if by_depth_bin:
        experiment_meta = _get_experiment_meta_dataframe()
        experiment_meta = _get_experiment_meta_with_depth_bin(
            experiment_meta=experiment_meta)
        cell_stats = cell_stats.merge(experiment_meta,
                                      on='experiment_id')
        g = sns.FacetGrid(cell_stats, col='depth_bin',
                          col_order=_get_sorted_depth_bin_labels(
                              experiment_meta=experiment_meta))
        g.map_dataframe(sns.histplot, x=stat_name, common_norm=False,
                        stat='percent', binwidth=0.05)
        g = _add_n_samples_to_title(
            fig=g,
            experiment_meta=experiment_meta[experiment_meta['experiment_id']
                .isin(targets['experiment_id'].unique())])
        g.fig.subplots_adjust(top=0.7)

        for i in range(len(g.axes[0])):
            g.axes[0, i].set_xlabel('Fraction of ROIs that are cell')
            g.axes[0, i].set_ylabel('Percent of FOV')
    else:
        title += f'\nR={cell_stats.shape[0]}'
        sns.histplot(cell_stats, x=stat_name, stat='percent')
    plt.suptitle(title)
    plt.show()


def _construct_dataset(
        user_labels: pd.DataFrame,
        job_regions: pd.DataFrame,
        only_false_negatives=False
):
    def _get_majority_label(labels):
        if labels.shape[0] == 1:
            label = labels['label'].iloc[0]
        else:
            label = \
                'cell' if labels['count'].iloc[0] > labels['count'].iloc[1] \
                    else 'not cell'
        return label

    user_labels = user_labels.merge(job_regions, left_on='region_id',
                                    right_on='id')
    user_labels['labels'] = user_labels['labels'].apply(
        lambda x: json.loads(x))

    region_ids = []
    experiment_ids = []
    user_ids = []
    for row in user_labels.itertuples(index=False):
        labels = pd.DataFrame(row.labels)
        if only_false_negatives:
            labels = labels[~labels['is_segmented']]
        else:
            labels = labels[labels['is_segmented']]

        if not labels.empty:
            region_ids += [row.region_id] * labels.shape[0]
            experiment_ids += [row.experiment_id] * labels.shape[0]
            user_ids += [row.user_id] * labels.shape[0]

    labels = pd.concat([pd.DataFrame(x) for x in user_labels['labels']],
                       ignore_index=True)

    if only_false_negatives:
        labels = labels[~labels['is_segmented']].copy()
    else:
        labels = labels[labels['is_segmented']].copy()

    labels['region_id'] = region_ids
    labels['experiment_id'] = experiment_ids
    labels['user_id'] = user_ids

    if not only_false_negatives:
        n_users_labeled = labels.groupby(['experiment_id', 'roi_id'])[
            'user_id'].nunique().reset_index().rename(
            columns={'user_id': 'n_users_labeled'})

        # Filter out ROIs with < 3 labels
        labels = labels.merge(n_users_labeled, on=['experiment_id', 'roi_id'])
        labels = labels[labels['n_users_labeled'] >= 3]
        labels = labels.drop(columns='n_users_labeled')

    majority_labels = labels.groupby(
        ['experiment_id', 'roi_id', 'label']).size().reset_index().rename(
        columns={0: 'count'}).sort_values('label').groupby(
        ['experiment_id', 'roi_id']).apply(_get_majority_label)
    majority_labels = majority_labels.reset_index()\
        .rename(columns={0: 'majority_label'})

    labels = labels.merge(majority_labels, on=['experiment_id', 'roi_id'])

    return labels


def _get_targets(labels: pd.DataFrame,
                 filter_by_maximally_labeled_fov=True):
    targets = labels.groupby(['experiment_id', 'roi_id'])['label'].apply(
        lambda x: 'cell' if len([l for l in x if l == 'cell']) >= 3
        else 'not cell')

    targets = targets.reset_index()

    targets = targets.merge(
        labels.drop_duplicates(subset=['experiment_id', 'roi_id'])
        [['experiment_id', 'roi_id', 'majority_label']],
        on=['experiment_id', 'roi_id'])

    if filter_by_maximally_labeled_fov:
        n_regions_per_exp = labels.groupby('experiment_id')[
            'region_id'].nunique()
        fully_labeled_exps = n_regions_per_exp[n_regions_per_exp == 3].index

        targets = targets[targets['experiment_id'].isin(fully_labeled_exps)]

    return targets


def _get_experiment_meta_with_depth_bin(experiment_meta: pd.DataFrame):
    exp_metas = []
    for row in experiment_meta.itertuples(index=False):
        exp_metas.append(ExperimentMetadata(
            experiment_id=row.experiment_id,
            imaging_depth=row.imaging_depth,
            equipment=row.equipment,
            problem_experiment=row.problem_experiment
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


def _get_n_in_depth_bin(experiment_meta: pd.DataFrame,
                        by_train_test_split=False):
    groupby = ['dataset', 'bin_id'] if by_train_test_split else ['bin_id']
    n_in_bin = experiment_meta.groupby(groupby).size().reset_index() \
        .rename(columns={0: 'n_in_bin'})
    return n_in_bin


def _get_sorted_depth_bin_labels(experiment_meta: pd.DataFrame):
    depth_bins = experiment_meta['depth_bin'].unique().tolist()
    depth_bins = sorted(depth_bins,
                        key=lambda x: int(re.findall(r'\[(\d+)', x)[0]))
    return depth_bins


def _add_n_samples_to_title(fig, experiment_meta: pd.DataFrame,
                            by_train_test_split=False,
                            title_prefix='N='):
    n_in_bin = _get_n_in_depth_bin(experiment_meta=experiment_meta,
                                   by_train_test_split=by_train_test_split)
    for i, row in enumerate(fig.axes):
        for j, col in enumerate(row):
            if by_train_test_split:
                if i == 0:
                    n = n_in_bin[n_in_bin['dataset'] == 'train'].iloc[j][
                        'n_in_bin']
                else:
                    n = n_in_bin[n_in_bin['dataset'] == 'test'].iloc[j][
                        'n_in_bin']
            else:
                n = n_in_bin.iloc[j]['n_in_bin']
            fig.axes[i][j].set_title(f'{col.get_title()}\n{title_prefix}{n}')
    return fig


def _get_experiment_meta_dataframe():
    experiment_meta = pd.read_json('~/Downloads/experiment_metadata.json')
    experiment_meta = experiment_meta.T

    with open('/Users/adam.amster/Downloads/experiment_metadata.json') as f:
        experiment_ids = json.load(f).keys()
    experiment_meta['experiment_id'] = experiment_ids
    experiment_meta['experiment_id'] = experiment_meta['experiment_id'] \
        .astype(int)
    experiment_meta = experiment_meta.reset_index(drop=True)
    return experiment_meta


def get_disagreement(
        user_labels: pd.DataFrame,
        job_regions: pd.DataFrame,
        by_depth=False,
        use_majority=False
):
    roi_label_agreement = _get_roi_label_agreement(user_labels=user_labels,
                                                   job_regions=job_regions,
                                                   use_majority=use_majority)

    fraction_groupby = ['fraction']
    if by_depth:
        experiment_meta = _get_experiment_meta_with_depth_bin(
            experiment_meta=_get_experiment_meta_dataframe()
        )
        roi_label_agreement = roi_label_agreement.merge(experiment_meta,
                                                        on='experiment_id')
        fraction_groupby.append('depth_bin')
    roi_agreement_fraction_counts = \
        roi_label_agreement.groupby(fraction_groupby).size()

    roi_agreement_fraction_counts = \
        roi_agreement_fraction_counts.reset_index()\
        .rename(columns={0: 'count', 'fraction': 'Cell agreement'})

    # Ignore all the outlier cases
    roi_agreement_fraction_counts = roi_agreement_fraction_counts[
        roi_agreement_fraction_counts['Cell agreement'].isin([0, 1/3, 2/3, 1])]

    roi_agreement_fraction_counts['Cell agreement'] = \
        roi_agreement_fraction_counts['Cell agreement']\
            .astype(str).apply(lambda x: x[:4])

    if by_depth:
        roi_agreement_fraction_counts['percent'] = \
            roi_agreement_fraction_counts.groupby(['depth_bin'])['count'].apply(
                lambda x: x / x.sum() * 100)
        g = sns.FacetGrid(roi_agreement_fraction_counts, col='depth_bin',
                          col_order=_get_sorted_depth_bin_labels(
                              experiment_meta=experiment_meta))
        g.map_dataframe(sns.barplot, x='Cell agreement', y='percent')
        g = _add_n_samples_to_title(fig=g,
                                    experiment_meta=roi_label_agreement)
        g.fig.subplots_adjust(top=0.7)
        plt.suptitle('Label Agreement by depth')
    else:
        roi_agreement_fraction_counts['percent'] = \
            roi_agreement_fraction_counts['count'] / \
            roi_agreement_fraction_counts['count'].sum() * 100
        sns.barplot(data=roi_agreement_fraction_counts, x='Cell agreement', y='percent')

        plt.title(f'Label Agreement\nN={roi_label_agreement.shape[0]}')
    plt.show()


def get_disagreement_by_user(user_labels: pd.DataFrame,
                             job_regions: pd.DataFrame):
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions)
    targets = _get_targets(
        labels=_construct_dataset(user_labels=user_labels,
                                  job_regions=job_regions),
        filter_by_maximally_labeled_fov=False)
    labels = labels.merge(targets, on=['experiment_id', 'roi_id'],
                          suffixes=('_user', '_total'))

    cells = labels[labels['majority_label'] == 'cell']
    user_disagreement = cells.groupby('user_id').apply(
        lambda x: (x['label_user'] != x['majority_label']).mean())
    user_disagreement = user_disagreement.reset_index()\
        .rename(columns={0: 'fraction'})

    num_regions_labeled = labels.groupby('user_id')['region_id'].nunique()
    num_regions_labeled = num_regions_labeled.reset_index().rename(
        columns={'region_id': 'num_regions'})
    user_disagreement = user_disagreement.merge(num_regions_labeled, on='user_id')
    user_disagreement = user_disagreement.sort_values('fraction')
    user_disagreement['Number of regions labeled'] = \
        pd.cut(user_disagreement['num_regions'], bins=4, precision=0)
    g = sns.barplot(data=user_disagreement, x='user_id', y='fraction',
                    hue='Number of regions labeled')
    g.get_xaxis().set_ticks([])

    plt.title('Fraction of times user disagreed with majority')
    plt.ylabel('Fraction')
    plt.xlabel('Labeler')
    plt.show()


def get_false_negatives(user_labels: pd.DataFrame, job_regions: pd.DataFrame):
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions,
                                only_false_negatives=True)
    targets = _get_targets(
        labels=_construct_dataset(user_labels=user_labels,
                                  job_regions=job_regions),
        filter_by_maximally_labeled_fov=False)

    print(labels.shape[0])
    print(labels.groupby('experiment_id')['user_id'].nunique())

    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=_get_experiment_meta_dataframe()
    )

    labels = labels.merge(experiment_meta, on='experiment_id')

    # get recall by depth bin
    targets = targets.merge(experiment_meta, on='experiment_id')
    targets = targets[targets['label'] == 'cell']
    tp = targets.groupby('depth_bin').size()
    fn = labels.groupby('depth_bin').size()
    recall = tp / (tp + fn)
    recall = recall.fillna(1.0)

    recall = recall.loc[
        _get_sorted_depth_bin_labels(experiment_meta=experiment_meta)]
    recall = recall.reset_index().rename(columns={0: 'sensitivity'})

    sns.barplot(data=recall, x='depth_bin', y='sensitivity')
    plt.title('Sensitivity by depth bin')
    plt.show()


def get_notes(job_region: pd.DataFrame):
    notes = pd.read_csv('~/Downloads/user_notes.csv')
    notes = notes.merge(job_region, left_on='region_id', right_on='id')
    text = notes['notes'].str.lower()
    text = text[~text.isna()]
    text = text.str.cat(sep=' ')

    # These are too prevalent and not informative. Removing
    text = text.replace('cells', '')
    text = text.replace('cell', '')
    text = text.replace('rois', '')
    text = text.replace('roi', '')

    wc = WordCloud()
    wc.generate_from_text(text=text)
    wc.to_file('/Users/adam.amster/Downloads/user_notes_wordcloud.png')

    roi_note_counts = notes.groupby(['experiment_id', 'roi_id']).size()

    notes = notes.set_index(['experiment_id', 'roi_id'])
    notes = notes.sort_index()
    notes = notes[~notes['notes'].isna()]
    print(notes[roi_note_counts >= 2][['notes', 'user_id']].drop_duplicates())
    print(notes[notes['notes'].str.lower().str.contains('island')
          ][['notes', 'user_id']])


def get_legacy_classifier_performance(user_labels: pd.DataFrame,
                                      job_regions: pd.DataFrame,
                                      by_depth=False,
                                      use_majority_label=False):
    label = 'majority_label' if use_majority_label else 'label'
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions)
    targets = _get_targets(labels=labels,
                           filter_by_maximally_labeled_fov=False)
    targets['experiment_id'] = targets['experiment_id'].astype(str)

    classifier_scores = _get_classifier_scores()
    classifier_scores = classifier_scores.rename(columns={'roi-id': 'roi_id'})

    res = classifier_scores.merge(targets, on=['experiment_id', 'roi_id'])
    res['y_pred'] = res['y_pred'].apply(lambda x: 'cell' if x else 'not cell')
    res = res.rename(columns={'y_score': 'Classifier score'})

    if by_depth:
        experiment_meta = _get_experiment_meta_with_depth_bin(
            experiment_meta=_get_experiment_meta_dataframe()
        )

        experiment_meta['experiment_id'] = experiment_meta['experiment_id'].astype(str)
        res = res.merge(experiment_meta[['experiment_id', 'depth_bin']],
                        on='experiment_id')
        g = sns.FacetGrid(res, col='depth_bin',
                          col_order=_get_sorted_depth_bin_labels(
                              experiment_meta=experiment_meta))
        g.map_dataframe(sns.histplot, x='Classifier score', hue='label',
                        multiple='dodge', stat='probability',
                        common_norm=False, hue_order=['not cell', 'cell'],
                        binwidth=0.05)
        g = _add_n_samples_to_title(
            fig=g,
            experiment_meta=targets.merge(experiment_meta, on='experiment_id'))
        g.fig.subplots_adjust(top=0.7)
        plt.suptitle('Legacy classifier score distribution by depth')

        # Since precision is problematic, show precision by depth bin
        print(res.groupby(['depth_bin']).apply(
            lambda x: precision_score(y_true=x[label], y_pred=x['y_pred'],
                                      pos_label='cell')))
        plt.show()

        # confusion matrices
        f, axes = plt.subplots(2, 6, figsize=(30, 10), sharey='row')

        for i, normalization in enumerate(('pred', 'true')):
            for j, depth_bin in enumerate(
                    # make depth bins sorted ascending
                    [x for x in reversed(res['depth_bin'].unique())]):
                df = res[res['depth_bin'] == depth_bin]
                disp = ConfusionMatrixDisplay.from_predictions(
                    y_true=df[label],
                    y_pred=df['y_pred'],
                    normalize=normalization)
                disp.plot(ax=axes[i, j])
                disp.ax_.set_title(
                    f'{depth_bin}\nN={df.shape[0]} '
                    f'normalized by {normalization}')
                disp.ax_.grid(False)
        plt.show()
    else:
        sns.histplot(res, x='Classifier score', hue='label', multiple='dodge',
                     stat='probability', common_norm=False)
        plt.title('Legacy classifier score distribution')
        plt.show()

        ConfusionMatrixDisplay.from_predictions(y_true=res[label],
                                                y_pred=res['y_pred'])
        plt.show()

    precision = precision_score(y_true=res[label], y_pred=res['y_pred'],
                                pos_label='cell')
    recall = recall_score(y_true=res[label], y_pred=res['y_pred'],
                          pos_label='cell')
    print(recall, precision)

    # Get misclassified examples
    print(res[res[label] == 'not cell']\
          .sort_values('Classifier score', ascending=False)
          [['experiment_id', 'roi_id', 'Classifier score']])

    # Get disagreement with examples with high classifier score
    agreement = _get_roi_label_agreement(user_labels=user_labels,
                                         job_regions=job_regions)
    agreement['experiment_id'] = agreement['experiment_id'].astype(str)
    agreement = agreement[['experiment_id', 'roi_id', 'fraction']]
    res = res.merge(agreement, on=['experiment_id', 'roi_id'])

    # Plot distribution of classifier score at varying levels of agreement
    sns.boxplot(data=res[res[label] == 'not cell'], x='fraction',
                y='Classifier score')
    plt.xlabel('Cell agreement')
    plt.title('Legacy classifier score at varying cell agreement')
    plt.show()


def get_overlabeled_rois(user_labels: pd.DataFrame, job_regions: pd.DataFrame):
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions)
    more_than_1_region = (labels.groupby(['experiment_id', 'roi_id'])[
                              'region_id'].nunique() > 1).reset_index().rename(
        columns={'region_id': 'more_than_one_region'})
    print(more_than_1_region['more_than_one_region'].mean())

    targets = labels.groupby(['region_id', 'roi_id'])['label'].apply(
        lambda x: 'cell' if len([l for l in x if l == 'cell']) >= 3
        else 'not cell').reset_index()
    targets['experiment_id'] = targets['region_id'].map(
        labels.set_index('region_id')['experiment_id'].to_dict())
    targets = targets.merge(more_than_1_region, on=['experiment_id', 'roi_id'])

    more_than_1_label = \
        (targets.groupby(['experiment_id', 'roi_id'])['label'].nunique() > 1)\
        .reset_index().rename(columns={'label': 'more_than_one_label'})
    targets = targets.merge(more_than_1_label, on=['experiment_id', 'roi_id'])

    print(targets[targets['more_than_one_region']]['more_than_one_label'].mean())


def _get_classifier_scores():
    preds_path = Path('/Users/adam.amster/Downloads/legacy_classifier_scores/')
    predictions = []
    for file in os.listdir(preds_path):
        predictions.append(
            pd.read_csv(preds_path / file, dtype={'experiment_id': str})
        )
    predictions = pd.concat(predictions, ignore_index=True)
    return predictions


def _get_roi_label_agreement(user_labels: pd.DataFrame,
                             job_regions: pd.DataFrame):
    label = 'label'
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions)
    roi_label_counts = labels.groupby(
        ['experiment_id', 'roi_id', label]).size()

    roi_label_agreement = roi_label_counts\
        .reset_index() \
        .rename(columns={0: 'count'})

    roi_label_agreement['fraction'] = \
        roi_label_agreement['count'] / 3

    # Account for cases with 0 cells
    zero_cells = \
        roi_label_agreement[(roi_label_agreement[label] == 'not cell') & (
                roi_label_agreement['count'] >= 3)].copy()
    zero_cells[label] = 'cell'
    zero_cells['fraction'] = 0.0
    roi_label_agreement = pd.concat([roi_label_agreement, zero_cells])

    roi_label_agreement = \
        roi_label_agreement[roi_label_agreement[label] == 'cell']

    # TODO why are there duplicates?
    roi_label_agreement = \
        roi_label_agreement.drop_duplicates(subset=['experiment_id', 'roi_id'])
    return roi_label_agreement


def get_train_test_split(user_labels: pd.DataFrame, job_regions: pd.DataFrame):
    labels = _construct_dataset(user_labels=user_labels,
                                job_regions=job_regions)

    experiment_meta = _get_experiment_meta_with_depth_bin(
        experiment_meta=_get_experiment_meta_dataframe()
    )
    targets = _get_targets(labels=labels,
                           filter_by_maximally_labeled_fov=False)
    experiment_meta = experiment_meta[experiment_meta['experiment_id']
        .isin(labels['experiment_id'].unique())]

    train_exp, test_exp = _get_train_test_experiments(
        experiment_metadata=experiment_meta)

    targets['dataset'] = \
        targets['experiment_id'].apply(
            lambda x: 'train' if x in train_exp else 'test')

    targets = targets.merge(experiment_meta, on='experiment_id')

    train = targets[targets['dataset'] == 'train']
    test = targets[targets['dataset'] == 'test']

    for plot in ('ROI', 'fov'):
        if plot == 'ROI':
            all_func = lambda x: x.shape[0] / targets.shape[0]
            train_func = lambda x: x.shape[0] / train.shape[0]
            test_func = lambda x: x.shape[0] / test.shape[0]
        else:
            all_func = lambda x: x['experiment_id'].nunique() / \
                                 targets['experiment_id'].nunique()
            train_func = lambda x: x['experiment_id'].nunique() / \
                                 train['experiment_id'].nunique()
            test_func = lambda x: x['experiment_id'].nunique() / \
                                 test['experiment_id'].nunique()

        all_depth_frac = targets.groupby('depth_bin').apply(
            all_func).reset_index().rename(
            columns={0: 'fraction'})
        all_depth_frac['dataset'] = 'all'

        train_depth_frac = train.groupby('depth_bin').apply(
            train_func).reset_index().rename(
            columns={0: 'fraction'})
        train_depth_frac['dataset'] = 'train'

        test_depth_frac = test.groupby('depth_bin').apply(
            test_func).reset_index().rename(
            columns={0: 'fraction'})
        test_depth_frac['dataset'] = 'test'

        depth_frac = pd.concat([all_depth_frac, train_depth_frac, test_depth_frac])

        sns.barplot(data=depth_frac, x='depth_bin', y='fraction', hue='dataset',
                    order=_get_sorted_depth_bin_labels(
                        experiment_meta=experiment_meta))
        plt.title(f'{plot} depth distribution across train/test split')

        plt.show()

    # How many regions per experiment in test
    targets = labels.groupby(['region_id', 'roi_id'])['label'].apply(
        lambda x: 'cell' if len([l for l in x if l == 'cell']) >= 3
        else 'not cell').reset_index()
    targets['experiment_id'] = targets['region_id'].map(
        labels.set_index('region_id')['experiment_id'].to_dict())
    targets['dataset'] = \
        targets['experiment_id'].apply(
            lambda x: 'train' if x in train_exp else 'test')
    test = targets[targets['dataset'] == 'test']
    region_per_fov = \
        test.groupby('experiment_id')['region_id'].nunique()\
        .reset_index()\
        .rename(columns={'region_id': 'count'})

    region_per_fov = region_per_fov.merge(experiment_meta, on='experiment_id')
    g = sns.FacetGrid(region_per_fov, col='depth_bin',
                      col_order=_get_sorted_depth_bin_labels(
                          experiment_meta=experiment_meta))
    g.map_dataframe(sns.histplot, x="count")

    g.fig.subplots_adjust(top=0.7)

    plt.suptitle('Number of regions labeled in test FOV')

    for i in range(len(g.axes[0])):
        g.axes[0, i].set_xlabel('Number of thrice labeled regions')
        g.axes[0, i].set_ylabel('Number of FOV')

    plt.show()


def main():
    user_labels = pd.read_csv('~/Downloads/user_labels.csv',
                              parse_dates=['timestamp'])
    job_regions = pd.read_csv('~/Downloads/job_regions.csv')

    # plot_duration_trends()
    # plot_distribution_of_depths(user_labels=user_labels)
    # get_overall_target_distribution(user_labels=user_labels,
    #                                 job_regions=job_regions)
    # get_cell_count_distribution(user_labels=user_labels, job_regions=job_regions)
    # get_cell_count_distribution(user_labels=user_labels, job_regions=job_regions,
    #                             by_depth_bin=True)
    # get_cell_count_distribution(user_labels=user_labels, job_regions=job_regions,
    #                             by_depth_bin=True,
    #                             cell_probability=True)
    # get_disagreement(user_labels=user_labels, job_regions=job_regions)
    # get_disagreement(user_labels=user_labels, job_regions=job_regions, use_majority=True)
    # get_disagreement(user_labels=user_labels, job_regions=job_regions,
    #                          by_depth=True)
    # get_false_negatives(user_labels=user_labels, job_regions=job_regions)
    # get_notes(job_region=job_regions)
    # get_legacy_classifier_performance(user_labels=user_labels,
    #                                   job_regions=job_regions)
    # get_legacy_classifier_performance(user_labels=user_labels,
    #                                   job_regions=job_regions,
    #                                   use_majority_label=True)
    # get_legacy_classifier_performance(user_labels=user_labels,
    #                                   job_regions=job_regions,
    #                                   by_depth=True)
    # get_legacy_classifier_performance(user_labels=user_labels,
    #                                   job_regions=job_regions,
    #                                   by_depth=True,
    #                                   use_majority_label=True)
    # get_overlabeled_rois(user_labels=user_labels, job_regions=job_regions)
    get_train_test_split(user_labels=user_labels, job_regions=job_regions)
    # get_disagreement_by_user(user_labels=user_labels, job_regions=job_regions)

if __name__ == '__main__':
    main()