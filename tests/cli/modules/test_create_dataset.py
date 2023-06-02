import random
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
        cls.roi_meta_dir = tempfile.TemporaryDirectory()

        cls.n_experiments = 2
        cls.experiment_ids = list(range(cls.n_experiments))
        cls.n_rois = 4
        cls.exp_metas = [{
                'experiment_id': str(idx),
                'imaging_depth': 1,
                'equipment': '2P',
                'problem_experiment': False
        }
            for idx in range(cls.n_experiments)]
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
            with open(Path(cls.roi_meta_dir.name) /
                      f'roi_meta_{exp_id}.json', 'w') as f:
                roi_meta = {
                    str(i): {
                        'is_inside_motion_border': random.choice([True, False])
                    }
                    for i in range(cls.n_rois)}
                f.write(json.dumps(roi_meta, indent=2))

    @classmethod
    def teardown_class(cls):
        cls.artifact_dir.cleanup()
        cls.roi_meta_dir.cleanup()

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
    @patch('deepcell.cli.modules.create_dataset._get_experiment_metadata')
    @pytest.mark.parametrize('include_only_rois_inside_motion_border',
                             (False, True))
    def test_create_train_test_split(
            self,
            mock_get_experiment_metadata,
            mock_get_raw_user_labels,
            include_only_rois_inside_motion_border
    ):
        """Test that the class loads the data and labels cells properly.
        """
        args = {
            'channels': [
                Channel.MASK.value,
                Channel.MAX_PROJECTION.value,
                Channel.CORRELATION_PROJECTION.value
            ],
            'cell_labeling_app_host': 'foo',
            'lims_db_username': 'foo',
            'lims_db_password': 'foo',
            "artifact_dir": {
                str(exp_id): str(self.artifact_dir.name)
                for exp_id in self.experiment_ids},
            "exp_roi_meta_path_map": {
                str(exp_id): str(Path(self.roi_meta_dir.name) /
                                 f'roi_meta_{exp_id}.json')
                for exp_id in self.experiment_ids},
            "min_labelers_required_per_region": 2,
            "vote_tallying_strategy": 'consensus',
            "test_size": 0.50,
            "n_depth_bins": 1,
            "seed": 1234,
            "output_dir": str(self.artifact_dir.name),
            "include_only_rois_inside_motion_border": (
                include_only_rois_inside_motion_border)
        }
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
        mock_get_experiment_metadata.return_value = self.exp_metas

        train_test = CreateDataset(args=[], input_data=args)
        train_test.run()

        with open(Path(self.artifact_dir.name) / "train_rois.json") as jfile:
            train_rois = json.load(jfile)
        with open(Path(self.artifact_dir.name) / "test_rois.json") as jfile:
            test_rois = json.load(jfile)

        if include_only_rois_inside_motion_border:
            with open(Path(self.roi_meta_dir.name) / 'roi_meta_0.json') as f:
                rois_meta = json.load(f)

            expected_train_rois = [
                roi_id for roi_id, roi_meta in rois_meta.items()
                if roi_meta['is_inside_motion_border']]

            with open(Path(self.roi_meta_dir.name) / 'roi_meta_1.json') as f:
                rois_meta = json.load(f)

            expected_test_rois = [
                roi_id for roi_id, roi_meta in rois_meta.items()
                if roi_meta['is_inside_motion_border']]
        else:
            expected_train_rois = [x['roi_id'] for x in train_rois]
            expected_test_rois = [x['roi_id'] for x in test_rois]

        assert len(train_rois) == len(expected_train_rois)
        for roi_id in expected_train_rois:
            assert roi_id in [x['roi_id'] for x in train_rois]
        assert all([roi['experiment_id'] == '0' for roi in train_rois])

        assert len(test_rois) == len(expected_test_rois)
        for roi_id in expected_test_rois:
            assert roi_id in [x['roi_id'] for x in test_rois]
        assert all([roi['experiment_id'] == '1' for roi in test_rois])

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
