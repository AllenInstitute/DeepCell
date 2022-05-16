import pandas as pd
import pytest
from flask import Flask
import json
from pathlib import Path
import shutil
from sqlalchemy import desc
import tempfile
from typing import Optional

from cell_labeling_app.database.database import db
from cell_labeling_app.database.populate_labeling_job import Region
from cell_labeling_app.database.schemas import (
    JobRegion, LabelingJob, User, UserLabels)

from deepcell.cli.modules.create_train_test_split import \
    CreateTrainTestSplit, \
    CreateTrainTestSplitInputSchema


class TestTrainTestSplitCli:

    @classmethod
    def setup_class(cls):
        file_loc = Path(__file__)
        cls.test_resource_dir = file_loc.parent / 'resources'
        cls.artifact_dir = tempfile.TemporaryDirectory()
        artifact_path = Path(cls.artifact_dir.name)
        cls.db_fp = tempfile.NamedTemporaryFile('w', suffix='.db')

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

        for exp_id in cls.exp_metas.keys():
            for roi_id in range(cls.n_rois):
                shutil.copyfile(
                    cls.test_resource_dir / "avg_12345_1.png",
                    artifact_path / f"avg_{exp_id}_{roi_id}.png")
                shutil.copyfile(
                    cls.test_resource_dir / "correlation_12345_1.png",
                    artifact_path / f"corr_{exp_id}_{roi_id}.png")
                shutil.copyfile(
                    cls.test_resource_dir / "mask_12345_1.png",
                    artifact_path / f"mask_{exp_id}_{roi_id}.png")
                shutil.copyfile(
                    cls.test_resource_dir / "max_12345_1.png",
                    artifact_path / f"max_{exp_id}_{roi_id}.png")

    def teardown(self):
        self.db_fp.close()
        self.artifact_dir.cleanup()

    def _init_db(self,
                 labels_per_region_limit: Optional[int] = None,
                 num_regions=2):
        app = Flask(__name__)
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{self.db_fp.name}'
        app.config['LABELERS_REQUIRED_PER_REGION'] = labels_per_region_limit
        db.init_app(app)
        with app.app_context():
            db.create_all()
        app.app_context().push()

        self.app = app

        self._populate_users()
        self._populate_labeling_job(num_regions=num_regions)

    def _populate_users(self):
        for user_id in self.user_region_ids:
            user = User(id=str(user_id))
            db.session.add(user)
            db.session.commit()

    @staticmethod
    def _populate_labeling_job(num_regions=2):
        job = LabelingJob()
        db.session.add(job)

        job_id = db.session.query(LabelingJob.job_id).order_by(desc(
            LabelingJob.date)).first()[0]

        for region_id in range(num_regions):
            region = Region(
                x=0,
                y=0,
                experiment_id=str(region_id),
                width=10,
                height=10
            )
            job_region = JobRegion(job_id=job_id,
                                   experiment_id=region.experiment_id,
                                   x=region.x, y=region.y, width=region.width,
                                   height=region.height)
            db.session.add(job_region)
        db.session.commit()

    @staticmethod
    def _add_labels(user_id: str,
                    region_id: int,
                    json_blob: str):
        user_labels = UserLabels(user_id=str(user_id),
                                 region_id=region_id,
                                 labels=json_blob)
        db.session.add(user_labels)
        db.session.commit()

    @staticmethod
    def _create_label_data(n_rois: int = 4,
                           cell_limit: int = 2,
                           add_point: bool = False):
        output = []
        for roi_idx in range(n_rois):
            label = "not cell"
            if roi_idx < cell_limit:
                label = "cell"
            output.append({"roi_id": roi_idx, "is_segmented": True,
                           "point": None, "label": label})
        if add_point:
            output.append({"roi_id": len(output) + 1, "is_segmented": False,
                           "point": (1, 2), "label": "cell"})
        return json.dumps(output)

    def test_create_train_test_split(self):
        """Test that the class loads the data and labels cells properly.
        """
        self._init_db()
        for region_id in self.user_region_ids:
            self._add_labels(
                user_id=0,
                region_id=region_id + 1,
                json_blob=self._create_label_data(n_rois=self.n_rois,
                                                  cell_limit=2,
                                                  add_point=True))
            self._add_labels(
                user_id=1,
                region_id=region_id + 1,
                json_blob=self._create_label_data(n_rois=self.n_rois,
                                                  cell_limit=1,
                                                  add_point=False))
        args = {"label_db": str(self.db_fp.name),
                "artifact_dir": str(self.artifact_dir.name),
                "experiment_metadata": str(self.exp_meta_file.name),
                "min_labelers_required_per_region": 2,
                "vote_threshold": 1,
                "test_size": 0.50,
                "n_depth_bins": 1,
                "seed": 1234,
                "output_dir": str(self.artifact_dir.name)}
        train_test = CreateTrainTestSplit(args=[], input_data=args)
        train_test.run()

        with open(Path(self.artifact_dir.name) / "train_rois.json") as jfile:
            train_rois = json.load(jfile)
        with open(Path(self.artifact_dir.name) / "test_rois.json") as jfile:
            test_rois = json.load(jfile)

        for roi_idx, roi in enumerate(train_rois):
            if roi_idx > 0:
                assert roi['label'] == 'not cell'
            else:
                assert roi['label'] == 'cell'
            assert roi['roi_id'] == str(roi_idx)
            assert roi['experiment_id'] == '0'

        for roi_idx, roi in enumerate(test_rois):
            if roi_idx > 0:
                assert roi['label'] == 'not cell'
            else:
                assert roi['label'] == 'cell'
            assert roi['roi_id'] == str(roi_idx)
            assert roi['experiment_id'] == '1'

    @pytest.mark.parametrize('labels, expected', (
            ([

                 pd.DataFrame({
                     'label': ['cell']
                 }),
                 pd.DataFrame({
                     'label': ['cell']
                 }),
                 pd.DataFrame({
                     'label': ['cell']
                 })
             ], 'cell'),
            ([
                 pd.DataFrame({
                     'label': ['cell']
                 }),
                 pd.DataFrame({
                     'label': ['cell']
                 }),
                 pd.DataFrame({
                     'label': ['not cell']
                 })
             ], 'cell'),
            ([
                 pd.DataFrame({
                     'label': ['not cell']
                 }),
                 pd.DataFrame({
                     'label': ['not cell']
                 }),
                 pd.DataFrame({
                     'label': ['not cell']
                 })
             ], 'not cell'),
            ([
                 pd.DataFrame({
                     'label': ['cell']
                 }),
                 pd.DataFrame({
                     'label': ['cell']
                 }),
                 pd.DataFrame({
                     'label': ['cell']
                 }),
                 pd.DataFrame({
                     'label': ['not cell']
                 }),
                 pd.DataFrame({
                     'label': ['not cell']
                 })
             ], 'cell')

    ))
    def test_tally_votes(self, labels, expected):
        assert CreateTrainTestSplitInputSchema._tally_votes(
            roi_id=0,
            region_labels=labels,
            vote_threshold=0.5
        ) == expected
