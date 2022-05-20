import json
import marshmallow
import pandas as pd
from pathlib import Path
import sqlalchemy as db
from typing import List

from argschema import ArgSchema, ArgSchemaParser, fields
from cell_labeling_app.database.schemas import JobRegion, UserLabels
from deepcell.data_splitter import DataSplitter
from deepcell.datasets.model_input import ModelInput
from deepcell.cli.schemas.data import ExperimentMetadataSchema


class CreateTrainTestSplitInputSchema(ArgSchema):
    label_db = fields.InputFile(
        required=True,
        description="Path to sqlite database containing all labels for the "
                    "app."
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

    @marshmallow.post_load
    def get_model_inputs(self, data):
        """Load the json blob labeling data from the app database and return
        the data as ModelInput objects.
        """
        model_inputs = []
        # Get all the potential regions.
        db_engine = db.create_engine(f"sqlite:///{data['label_db']}")
        region_query = db.select([JobRegion.id, JobRegion.experiment_id])
        region_data = db_engine.execute(region_query).fetchall()

        for (region_id, exp_id) in region_data:
            # For each region, load the user labels from the database.
            label_query = db.select([UserLabels.user_id,
                                     UserLabels.region_id,
                                     UserLabels.labels])
            label_query = label_query.where(
                UserLabels.region_id == region_id)
            label_results = db_engine.execute(label_query).fetchall()
            # Count the number of times this region was classified by a
            # User.
            n_labelers = len(label_results)
            if n_labelers < data["min_labelers_required_per_region"]:
                continue
            # Load the labeling data if the number of classifiers is
            # sufficient.
            region_labels = [pd.DataFrame(
                                 data=json.loads(labels)).set_index('roi_id')
                             for _, _, labels in label_results]
            for roi_id, roi_row in region_labels[0].iterrows():
                # Don't create a ModelInput if the label is a False Negative.
                if not roi_row['is_segmented']:
                    continue
                label = self._tally_votes(
                    roi_id=roi_id,
                    region_labels=region_labels,
                    vote_threshold=data['vote_threshold'])

                # Create the output model input for the ROI.
                model_input = ModelInput.from_data_dir(
                    data_dir=data['artifact_dir'],
                    experiment_id=str(exp_id),
                    roi_id=str(roi_id),
                    label=label)
                model_inputs.append(model_input)
        data['model_inputs'] = model_inputs
        return data

    @staticmethod
    def _tally_votes(roi_id: int,
                     region_labels: List[pd.DataFrame],
                     vote_threshold: float):
        """Loop through labels and return a label of "cell" or "not cell" given
        the votes of the labelers.

        Parameters
        ---------
        roi_id : int
            Id of the ROI to consider.
        region_labels : List[pandas.DataFrame]
            Set of labels for all ROIs from all labelers over the region.
            DataFrames must be indexed by ``roi_id`` values.
        vote_threshold :  float
            Fraction of votes for "cell" to return a label for the ROI of
            "cell".

        Returns
        -------
        label : str
            Label for ROI. "cell" or "not cell"
        """
        n_cell_votes = 0
        n_total_votes = 0
        label = 'not cell'
        for roi_labeler in region_labels:
            if roi_labeler.loc[roi_id, 'label'] == 'cell':
                n_cell_votes += 1
            n_total_votes += 1
        if n_cell_votes / n_total_votes > vote_threshold:
            label = 'cell'
        return label

    @marshmallow.post_load
    def get_experiment_metadata(self, data):
        """Load experiment metadata and trim to only those experiments that
        are labeled by users.
        """
        # Get the set of experiments that has a user label given.
        db_engine = db.create_engine(f"sqlite:///{data['label_db']}")
        exp_id_query = db.select(db.distinct(JobRegion.experiment_id))
        exp_id_query = exp_id_query.join(UserLabels,
                                         JobRegion.id == UserLabels.region_id)
        exp_id_data = db_engine.execute(exp_id_query).fetchall()

        # Retrieve the data for the experiments that have a user label.
        with open(data['experiment_metadata'], 'r') as f:
            exp_metas = json.load(f)
        output_exp_metas = []
        for (exp_id,) in exp_id_data:
            meta = exp_metas[exp_id]
            output_exp_metas.append(ExperimentMetadataSchema().load(
                dict(experiment_id=str(exp_id),
                     imaging_depth=meta['imaging_depth'],
                     equipment=meta['equipment'],
                     problem_experiment=meta['problem_experiment'])))
        data['experiment_metadata'] = output_exp_metas
        return data


class CreateTrainTestSplit(ArgSchemaParser):
    """Access the cell labeling app database and return ROIs in a train
    test split by experiment.

    Store the ROIs as json files.
    """
    default_schema = CreateTrainTestSplitInputSchema

    def run(self):
        model_inputs = self.args['model_inputs']

        splitter = DataSplitter(model_inputs=model_inputs,
                                seed=self.args['seed'])
        train_rois, test_rois = splitter.get_train_test_split(
            test_size=self.args['test_size'],
            full_dataset=model_inputs,
            exp_metas=self.args['experiment_metadata'],
            n_depth_bins=self.args['n_depth_bins'])

        train_dicts = [model_roi.to_dict()
                       for model_roi in train_rois.model_inputs]
        test_dicts = [model_roi.to_dict()
                      for model_roi in test_rois.model_inputs]

        output_dir = Path(self.args['output_dir'])
        with open(output_dir / 'train_rois.json', 'w') as jfile:
            json.dump(train_dicts, jfile)
        with open(output_dir / 'test_rois.json', 'w') as jfile:
            json.dump(test_dicts, jfile)


if __name__ == "__main__":
    create_test_train_split = CreateTrainTestSplit()
    create_test_train_split.run()
