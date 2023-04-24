import json
from typing import Optional

import pandas as pd
import requests

from deepcell.util.construct_dataset.vote_tallying_strategy import \
    VoteTallyingStrategy


def construct_dataset(
        cell_labeling_app_host: str,
        min_labelers_per_roi: int = 3,
        vote_tallying_strategy: Optional[VoteTallyingStrategy] = None,
        raw: bool = False
) -> pd.DataFrame:
    """
    Create labeled dataset from label db

    @param cell_labeling_app_host: Cell labeling app host
    @param min_labelers_per_roi: minimum number of labelers required to have
    seen a given ROI
    @param vote_tallying_strategy: VoteTallyingStrategy
    @param raw: If True, returns untallied labels
    @return: pd.DataFrame
    """
    if not raw and vote_tallying_strategy is None:
        raise ValueError('vote_strategy required if tallying votes')

    user_labels = _get_raw_user_labels(
        cell_labeling_app_host=cell_labeling_app_host,
    )
    user_labels['labels'] = user_labels['labels'].apply(
        lambda x: json.loads(x))

    experiment_ids = []
    user_ids = []

    def _get_is_user_added_mask(labels: pd.DataFrame):
        # is_segmented is old key, is_user_added is new one
        return ~labels['is_user_added'] if 'is_user_added' in labels else \
            labels['is_segmented']

    for row in user_labels.itertuples(index=False):
        labels = pd.DataFrame(row.labels)
        labels = labels[_get_is_user_added_mask(labels=labels)]

        if not labels.empty:
            experiment_ids += [row.experiment_id] * labels.shape[0]
            user_ids += [row.user_id] * labels.shape[0]

    labels = pd.concat([pd.DataFrame(x) for x in user_labels['labels']],
                       ignore_index=True)

    labels = labels[_get_is_user_added_mask(labels=labels)].copy()

    labels['experiment_id'] = experiment_ids
    labels['user_id'] = user_ids

    n_users_labeled = labels.groupby(['experiment_id', 'roi_id'])[
        'user_id'].nunique().reset_index().rename(
        columns={'user_id': 'n_users_labeled'})

    # Filter out ROIs with < min_labelers_per_roi labels
    labels = labels.merge(n_users_labeled, on=['experiment_id', 'roi_id'])
    labels = labels[labels['n_users_labeled'] >= min_labelers_per_roi]
    labels = labels.drop(columns='n_users_labeled')

    # Drop cases where a single user labeled an ROI more than once
    labels = labels.drop_duplicates(
        subset=['experiment_id', 'roi_id', 'user_id'])

    if not raw:
        labels = tally_votes(labels=labels,
                             vote_tallying_strategy=vote_tallying_strategy)

    labels['experiment_id'] = labels['experiment_id'].astype(str)
    labels['roi_id'] = labels['roi_id'].astype(str)
    return labels


def tally_votes(labels: pd.DataFrame,
                vote_tallying_strategy: VoteTallyingStrategy):
    """

    @param labels: raw, untallied labels dataframe
    @param vote_tallying_strategy: VoteTallyingStrategy
    @return: labels dataframe with a single record per ROI with the resulting
        label using `vote_threshold`
    """
    labels = labels.groupby(['experiment_id', 'roi_id'])['label'] \
        .apply(lambda x: _tally_votes_for_observation(
               labels=x, vote_tallying_strategy=vote_tallying_strategy))
    labels = labels.reset_index()
    return labels


def _tally_votes_for_observation(
        labels: pd.Series,
        vote_tallying_strategy: VoteTallyingStrategy
):
    """Loop through labels and return a label of "cell" or "not cell" given
    the votes of the labelers.

    Parameters
    ---------
    labels : pandas.Series
        List of user labels
    vote_tallying_strategy :  VoteTallyingStrategy

    Returns
    -------
    label : str
        Label for ROI. "cell" or "not cell"
    """
    cell_count = len([label for label in labels if label == 'cell'])

    if vote_tallying_strategy == VoteTallyingStrategy.CONSENSUS:
        label = 'cell' if cell_count == len(labels) else 'not cell'
    elif vote_tallying_strategy == VoteTallyingStrategy.MAJORITY:
        label = 'cell' if cell_count / len(labels) > 0.5 else 'not cell'
    elif vote_tallying_strategy == VoteTallyingStrategy.ANY:
        label = 'cell' if cell_count > 0 else 'not cell'
    else:
        raise ValueError(f'vote strategy {vote_tallying_strategy} not '
                         f'supported')

    return label


def _get_raw_user_labels(
    cell_labeling_app_host: str,
) -> pd.DataFrame:
    """
    Queries label database for user labels
    @param cell_labeling_app_host: Cell labeling app host
    @return: pd.DataFrame
    """
    url = f'http://{cell_labeling_app_host}/get_all_labels'
    r = requests.get(url)
    labels = r.json()
    labels = pd.DataFrame(labels)
    return labels
