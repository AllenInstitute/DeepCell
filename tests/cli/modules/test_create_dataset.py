from unittest.mock import patch

import pandas as pd
import pytest
import json
from pathlib import Path
import tempfile


from deepcell.cli.modules.create_dataset import \
    CreateDataset, _tally_votes_for_observation, VoteTallyingStrategy
from deepcell.datasets.channel import Channel
from deepcell.testing.util import get_test_data


class TestTrainTestSplitCli:

    @classmethod
    def setup_class(cls):
        file_loc = Path(__file__)
        cls.test_resource_dir = file_loc.parent / 'resources'
        cls.artifact_dir = tempfile.TemporaryDirectory()

        cls.n_experiments = 2
        cls.n_rois = 4
        cls.exp_metas = dict([(str(idx), {'experiment_id': str(idx),
                                          'imaging_depth': 1,
                                          'equipment': '2P',
                                          'problem_experiment': False})
                              for idx in range(cls.n_experiments)])
        cls.exp_meta_file = tempfile.NamedTemporaryFile('w', suffix='.json')
        with open(cls.exp_meta_file.name, 'w') as jfile:
            json.dump(cls.exp_metas, jfile, indent=2)
        cls.user_region_ids = list(range(cls.n_experiments))

        for exp_id in range(cls.n_experiments):
            get_test_data(
                write_dir=cls.artifact_dir.name,
                exp_id=str(exp_id),
                n_rois=cls.n_rois
            )

    def teardown(self):
        self.artifact_dir.cleanup()

    @staticmethod
    def _create_label_data(n_rois: int = 4,
                           cell_limit: int = 2,
                           add_user_added_roi: bool = False):
        output = []
        for roi_idx in range(n_rois):
            label = "not cell"
            if roi_idx < cell_limit:
                label = "cell"
            output.append({"roi_id": roi_idx,
                           "is_user_added": False,
                           "label": label})
        if add_user_added_roi:
            output.append({
                'roi_id': len(output) + 1,
                'is_user_added': True,
                'label': 'cell'
            })
        return json.dumps(output)

    @patch('deepcell.cli.modules.create_dataset._get_raw_user_labels')
    def test_create_train_test_split(self, mock_get_raw_user_labels):
        """Test that the class loads the data and labels cells properly.
        """
        args = {
            'channels': [
                Channel.MASK.value,
                Channel.MAX_PROJECTION.value,
                Channel.CORRELATION_PROJECTION.value
            ],
            'cell_labeling_app_host': 'foo',
            "artifact_dir": str(self.artifact_dir.name),
            "experiment_metadata": str(self.exp_meta_file.name),
            "min_labelers_required_per_region": 2,
            "vote_tallying_strategy": 'consensus',
            "test_size": 0.50,
            "n_depth_bins": 1,
            "seed": 1234,
            "output_dir": str(self.artifact_dir.name)}
        labels = pd.DataFrame({
            'experiment_id': ['0', '0', '1', '1'],
            'user_id': [0, 1] * 2,
            'labels': [
                self._create_label_data(
                    n_rois=self.n_rois,
                    cell_limit=2,
                    add_user_added_roi=True
                ),
                self._create_label_data(
                    n_rois=self.n_rois,
                    cell_limit=1,
                    add_user_added_roi=False
                ),
                self._create_label_data(
                    n_rois=self.n_rois,
                    cell_limit=2,
                    add_user_added_roi=True
                ),
                self._create_label_data(
                    n_rois=self.n_rois,
                    cell_limit=1,
                    add_user_added_roi=False
                )
            ]
        })

        mock_get_raw_user_labels.return_value = labels

        train_test = CreateDataset(args=[], input_data=args)
        train_test.run()

        with open(Path(self.artifact_dir.name) / "train_rois.json") as jfile:
            train_rois = json.load(jfile)
        with open(Path(self.artifact_dir.name) / "test_rois.json") as jfile:
            test_rois = json.load(jfile)

        for roi_idx, roi in enumerate(train_rois):
            assert roi['roi_id'] == str(roi_idx)
            assert roi['experiment_id'] == '0'

        for roi_idx, roi in enumerate(test_rois):
            assert roi['roi_id'] == str(roi_idx)
            assert roi['experiment_id'] == '1'

    @pytest.mark.parametrize('labels, expected', (
            (pd.Series(['cell', 'cell', 'cell']), ('cell', 'cell', 'cell')),
            (pd.Series(['cell', 'cell', 'not cell']),
             ('cell', 'not cell', 'cell')),
            (pd.Series(['not cell' 'not cell', 'not cell']),
             ('not cell', 'not cell', 'not cell')),
            (pd.Series(['cell', 'cell', 'cell', 'not cell', 'not cell']),
             ('cell', 'not cell', 'cell'))

    ))
    def test_tally_votes(self, labels, expected):
        for i, vote_tallying_strategy in enumerate(
                (VoteTallyingStrategy.MAJORITY, VoteTallyingStrategy.CONSENSUS,
                 VoteTallyingStrategy.ANY)):
            assert _tally_votes_for_observation(
                labels=labels,
                vote_tallying_strategy=vote_tallying_strategy
            ) == expected[i]
