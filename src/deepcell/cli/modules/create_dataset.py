import json
from typing import Optional, List, Dict

import marshmallow
import pandas as pd
from pathlib import Path
import sqlalchemy as db

from argschema import ArgSchema, ArgSchemaParser, fields
from cell_labeling_app.database.schemas import JobRegion, UserLabels
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput
from deepcell.cli.schemas.data import ExperimentMetadataSchema


class CreateDatasetInputSchema(ArgSchema):
    labels_path = fields.InputFile(
        required=True,
        description="Path to labels. Must either be sqlite database "
                    "containing all labels or already preprocessed labels "
                    "in csv format"
    )
    artifact_dir = fields.InputDir(
        required=True,
        description="Directory containing image artifacts for all labeled "
                    "ROIs.",
    )
    experiment_metadata = fields.InputFile(
        required=True,
        description="File containing metadata relating the the experiments "
                    "from which the labeled ROIs were drawn."
    )
    min_labelers_required_per_region = fields.Int(
        required=True,
        default=3,
        description="Minimum number of labelers that have looked at an ROI "
                    "to consider the label robust.",
    )
    vote_threshold = fields.Float(
        required=True,
        default=0.5,
        description="Fraction of labelers that must agree on a label for the "
                    "ROI to be considered a 'cell'",
    )
    test_size = fields.Float(
        required=True,
        default=0.2,
        description="Fraction of the experiments to reserve for the test "
                    "sample. Default: 0.2"
    )
    n_depth_bins = fields.Int(
        required=True,
        default=5,
        description="Number of depth bins to divide experiments into."
    )
    seed = fields.Int(
        required=False,
        allow_none=True,
        default=None,
        descripion="Integer seed to feed to the test/train splitter."
    )
    output_dir = fields.OutputDir(
        required=True,
        description="Output directory to write out json files containing the "
                    "training and test datasets.",
    )


class CreateDataset(ArgSchemaParser):
    """Convert a set of labeled ROIs into a dataset usable by the model
    Splits data into a train and test set and outputs both as json files.
    """
    default_schema = CreateDatasetInputSchema

    def run(self):
        labels_path = Path(self.args['labels_path'])
        output_dir = Path(self.args['output_dir'])

        if labels_path.suffix not in ('.db', '.csv'):
            raise ValueError(f'Expected labels_path to be a db or csv but got '
                             f'file with suffix {labels_path.suffix}')

        if labels_path.suffix == '.db':
            raw_labels = construct_dataset(
                label_db_path=self.args['labels_path'],
                min_labelers_per_roi=(
                    self.args['min_labelers_required_per_region']),
                raw=True)
        else:
            raw_labels = pd.read_csv(self.args['labels_path'],
                                 dtype={'experiment_id': str, 'roi_id': str})
        raw_labels = raw_labels[['experiment_id', 'roi_id', 'user_id', 'label']]

        # Anonymize user id
        raw_labels['user_id'] = raw_labels['user_id'].map(
            {user_id: i for i, user_id in
             enumerate(raw_labels['user_id'].unique())})

        raw_labels.to_csv(output_dir / 'raw_labels.csv', index=False)
        
        labels = _tally_votes(labels=raw_labels,
                              vote_threshold=self.args['vote_threshold'])

        model_inputs = [
            ModelInput.from_data_dir(
                data_dir=self.args['artifact_dir'],
                experiment_id=row.experiment_id,
                roi_id=row.roi_id,
                label=row.label
            )
            for row in labels.itertuples(index=False)
        ]

        experiment_metadata = \
            _get_experiment_metadata(
                experiment_ids=labels['experiment_id'].unique(),
                experiment_metadata_path=self.args['experiment_metadata'])

        splitter = DataSplitter(model_inputs=model_inputs,
                                seed=self.args['seed'])
        train_rois, test_rois = splitter.get_train_test_split(
            test_size=self.args['test_size'],
            full_dataset=model_inputs,
            exp_metas=experiment_metadata,
            n_depth_bins=self.args['n_depth_bins'])

        train_dicts = [model_roi.to_dict()
                       for model_roi in train_rois.model_inputs]
        test_dicts = [model_roi.to_dict()
                      for model_roi in test_rois.model_inputs]

        with open(output_dir / 'train_rois.json', 'w') as jfile:
            json.dump(train_dicts, jfile)
        with open(output_dir / 'test_rois.json', 'w') as jfile:
            json.dump(test_dicts, jfile)


def construct_dataset(
        label_db_path: str,
        min_labelers_per_roi: int,
        vote_threshold: Optional[float] = None,
        raw: bool = False
) -> pd.DataFrame:
    """
    Create labeled dataset from label db

    @param label_db_path: Path to label database
    @param min_labelers_per_roi: minimum number of labelers required to have
    seen a given ROI
    @param vote_threshold: Threshold to mark an ROI as cell
    @param raw: If True, returns untallied labels
    @return: pd.DataFrame
    """
    if not raw and vote_threshold is None:
        raise ValueError('vote_threshold required if tallying votes')

    user_labels = _get_raw_user_labels(label_db_path=label_db_path)
    user_labels['labels'] = user_labels['labels'].apply(
        lambda x: json.loads(x))

    experiment_ids = []
    user_ids = []
    for row in user_labels.itertuples(index=False):
        labels = pd.DataFrame(row.labels)
        labels = labels[labels['is_segmented']]

        if not labels.empty:
            experiment_ids += [row.experiment_id] * labels.shape[0]
            user_ids += [row.user_id] * labels.shape[0]

    labels = pd.concat([pd.DataFrame(x) for x in user_labels['labels']],
                       ignore_index=True)

    labels = labels[labels['is_segmented']].copy()

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
        labels = _tally_votes(labels=labels, vote_threshold=vote_threshold)

    labels['experiment_id'] = labels['experiment_id'].astype(str)
    labels['roi_id'] = labels['roi_id'].astype(str)
    return labels


def _tally_votes(labels: pd.DataFrame, vote_threshold: float):
    """

    @param labels: raw, untallied labels dataframe
    @param vote_threshold:
    @return: labels dataframe with a single record per ROI with the resulting
        label using `vote_threshold`
    """
    labels = labels.groupby(['experiment_id', 'roi_id'])['label'] \
        .apply(lambda x: _tally_votes_for_observation(
               labels=x, vote_threshold=vote_threshold))
    labels = labels.reset_index()
    return labels


def _tally_votes_for_observation(
        labels: pd.Series,
        vote_threshold: float
):
    """Loop through labels and return a label of "cell" or "not cell" given
    the votes of the labelers.

    Parameters
    ---------
    labels : pandas.Series
        List of user labels
    vote_threshold :  float
        Fraction of votes for "cell" to return a label for the ROI of
        "cell".

    Returns
    -------
    label : str
        Label for ROI. "cell" or "not cell"
    """
    cell_count = len([label for label in labels if label == 'cell'])
    cell_frac = cell_count / len(labels)
    return 'cell' if cell_frac >= vote_threshold else 'not cell'


def _get_raw_user_labels(label_db_path: str) -> pd.DataFrame:
    """
    Queries label database for user labels
    @param label_db_path: Path to label database
    @return: pd.DataFrame
    """
    db_engine = db.create_engine(f"sqlite:///{label_db_path}")
    query = (db.select([JobRegion.experiment_id,
                        UserLabels.labels,
                        UserLabels.user_id])
             .select_from(UserLabels).join(JobRegion,
                                           JobRegion.id ==
                                           UserLabels.region_id))
    labels = pd.read_sql(sql=query, con=db_engine)
    return labels

def _get_experiment_metadata(experiment_ids: List,
                             experiment_metadata_path: str) -> List[Dict]:
    """Load experiment metadata for `experiment_ids`
    @param experiment_ids: List of experiment ids to get metadata for
    @param experiment_metadata_path: Path to experiment metadata
    @return: List of experiment metadata
    """
    with open(experiment_metadata_path, 'r') as f:
        exp_metas = json.load(f)
    output_exp_metas = []
    for exp_id in experiment_ids:
        meta = exp_metas[exp_id]
        output_exp_metas.append(ExperimentMetadataSchema().load(
            dict(experiment_id=str(exp_id),
                 imaging_depth=meta['imaging_depth'],
                 equipment=meta['equipment'],
                 problem_experiment=meta['problem_experiment'])))
    return output_exp_metas

if __name__ == "__main__":
    create_test_train_split = CreateDataset()
    create_test_train_split.run()
