import logging
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Generator, Dict, Tuple, Set, Iterable, Union
from urllib.parse import urlparse

import boto3
import pandas as pd
from croissant.utils import read_jsonlines, json_load_local_or_s3
from tqdm import tqdm

from deepcell.datasets.artifact import Artifact

logging.basicConfig(level=logging.INFO)


class VisualBehaviorDataset:
    """Class representing the visual behavior dataset from AWS sagemaker
    labeling jobs"""
    def __init__(self, artifact_destination: Path,
                 exclude_projects: Optional[List[str]] = None,
                 debug=False):
        """
        Args:
            - exclude_projects:
                an optional list of project names to exclude from being
                included in the dataset

        """
        self._logger = logging.getLogger(__name__)
        self._exclude_projects = exclude_projects
        self._artifact_destination = artifact_destination

        # Due to historic reasons, some ROI ids are stored in the manifest
        # using a globally unique ID
        # This object maps from global to an experiment-local id and vice versa
        self._global_to_local_map = json_load_local_or_s3(
            uri='s3://prod.slapp.alleninstitute.org/visual_behavior'
                '/global_to_local_mapping.json')

        self._logger.info('Reading manifests and preprocessing')
        self._dataset, artifact_dirs, project_meta = self._get_dataset()

        self._debug = debug

        os.makedirs(self._artifact_destination, exist_ok=True)

        if debug:
            self._dataset = self._dataset[:2]
            self._download_files()
        else:
            self._logger.info('Downloading dataset')
            self._download_dataset(artifact_dirs=artifact_dirs)

        self._logger.info('Renaming artifacts')
        self._rename_artifacts()

        self._logger.info('Updating paths')
        self._update_artifact_locations_from_s3_to_local()

    @property
    def dataset(self) -> List[Artifact]:
        return self._dataset

    def _get_dataset(self) -> \
            Tuple[List[Artifact], Set[str], pd.DataFrame]:
        """
        Each labeling job on AWS sagemaker produces a manifest.
        This function combines those manifests into a single set of labeled
        records (List of Artifact).
        It handles duplicates, and any problem records (no label,
        failure-reason)
        It renames all roi ids using the convention
        exp_<experiment_id>_roi_<local_roi_id>

        Returns:
            Tuple -
                rois: The set of ROIs in the dataset, including metadata
                artifact_dirs: Which artifact dirs on s3 are being used
                project_counts: Count of examples in each labeling job
                project_name_meta: Metadata about each labeling job
        """

        def is_exclude_project(x: dict) -> bool:
            """Returns True if the project is in the exclusion list

            Args:
                - x: A manifest record

            Returns:
                True if the project is in the exclusion list, false otherwise
            """
            exclude = False
            for p in self._exclude_projects:
                if p in x:
                    exclude = True
                    break
            return exclude

        manifests = self._get_manifests()
        rois = []
        artifact_dirs = set()
        seen = set()
        project_name_meta = []
        for manifest in manifests:
            for x in manifest:
                if self._exclude_projects is not None:
                    if is_exclude_project(x=x):
                        continue

                metadata_key = [x for x in x.keys() if 'metadata' in x]
                if len(metadata_key) == 1:
                    if 'failure-reason' in x[metadata_key[0]]:
                        continue
                experiment_id = x['experiment-id']
                roi_id = x['roi-id']

                try:
                    roi_id = int(roi_id)
                except ValueError:
                    # ie 'exp_920211809_226'
                    roi_id = roi_id.split('_')[-1]

                if int(roi_id) >= 1000000:
                    # it is global, the globally unique ids start at 1000000
                    # and no experiment would have that many ROIs
                    roi_id = self._global_to_local_map['global_to_local'][str(
                        roi_id)]
                    roi_id = roi_id['roi_id']

                roi_id = f'exp_{experiment_id}_roi_{roi_id}'
                x['roi-id'] = roi_id

                global_id = self._global_to_local_map['local_to_global'][
                    roi_id]
                if global_id in self._global_to_local_map['degeneracies']:
                    # It is a duplicate
                    continue

                if roi_id in seen:
                    # also a duplicate
                    continue

                label = None
                project_name = None
                for key in x:
                    if isinstance(x[key], dict):
                        if 'majorityLabel' in x[key]:
                            label = x[key]['majorityLabel']
                            project_name = key
                            break
                if label is None:
                    continue

                try:
                    project_name_meta.append(
                        {'project_name': project_name,
                         'date': x[f'{project_name}-metadata'][
                             'creation-date']})
                except KeyError:
                    # if it doesn't have this metadata key, something is off
                    # about it
                    # skipping the record
                    continue
                artifact_dir = re.match(
                    r's3://prod\.slapp\.alleninstitute\.org/([\w\d\-_]+)/',
                    x['source-ref']).group()
                artifact_dirs.add(artifact_dir)

                artifact = Artifact(
                    roi_id=x['roi-id'],
                    experiment_id=x['experiment-id'],
                    max_projection_path=x['max-source-ref'],
                    avg_projection_path=x['avg-source-ref'],
                    mask_path=x['roi-mask-source-ref'],
                    label=label)
                rois.append(artifact)
                seen.add(roi_id)
        return rois, artifact_dirs, pd.DataFrame(project_name_meta)

    def _rename_artifacts(self):
        """
        Conform artifacts to <experiment_id>_<roi_id> convention
        Returns:
            None, renames files inplace
        """
        files = os.listdir(self._artifact_destination)
        for file in tqdm(files):
            file = Path(file)
            name = file.stem.split('_')
            suffix = file.suffix
            original_id = name[-1]
            try:
                int(original_id)
            except ValueError:
                # If not an int, just assume it is already in local format
                continue
            if int(original_id) >= 1000000:
                exp_id = self._global_to_local_map['global_to_local'][
                    original_id]['experiment_id']
                roi_id = self._global_to_local_map['global_to_local'][
                    original_id]['roi_id']
                new_id = f'exp_{exp_id}_roi_{roi_id}'

                original_path = self._artifact_destination / file
                new_path = self._artifact_destination / f'{name[0]}_' \
                                                        f'{new_id}{suffix}'
                cmd = f'mv {original_path} {new_path}'
                self._logger.debug(cmd)
                os.system(cmd)

    def _download_dataset(self, artifact_dirs: Iterable[str]):
        """Downloads artifacts given by artifact_dirs"""
        for bucket in artifact_dirs:
            with tempfile.TemporaryDirectory() as d:
                cmd = f'aws s3 sync {bucket} {d} --exclude "*" ' \
                      f'--include "*.png"'
                os.system(cmd)

                # s3 sync downloads files include directory structure
                # Remove directory structure and just copy files to destination
                os.system(f'mv {d}/*/*.png {self._artifact_destination}')

    def _download_files(self):
        """Use for debugging. Downloads select files. (slow)"""
        obj_files_map = defaultdict(list)

        for x in self._dataset:
            artifacts = [x.max_projection_path, x.avg_projection_path,
                         x.mask_path]
            for artifact in artifacts:
                parsed = urlparse(str(artifact))
                obj = Path(parsed.path)
                obj = str(obj).replace(obj.name, '')
                obj_files_map[obj].append(Path(parsed.path).name)

        for obj, files in obj_files_map.items():
            for file in files:
                cmd = f'aws s3 cp --recursive ' \
                      f's3://prod.slapp.alleninstitute.org{obj}' \
                      f' {self._artifact_destination}' \
                      f' --exclude "*" --include "{file}"'
                self._logger.debug(cmd)
                os.system(cmd)

    @staticmethod
    def _get_manifests() -> List[Generator[Dict, None, None]]:
        """
        Each labeling job on AWS sagemaker produces a manifest
        This function retrieves all manifests for visual behavior labeling jobs

        Returns:
            List of generator of manifest records
        """
        s3 = boto3.client('s3')
        bucket = 'prod.slapp.alleninstitute.org'

        prefix = 'visual_behavior/manifests/'
        objs = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix)
        files = [obj['Key'] for obj in objs['Contents'] if
                 obj['Key'] != prefix]
        manifests = [read_jsonlines(uri=f's3://{bucket}/{file}') for file in
                     files]
        return manifests

    def _update_artifact_locations_from_s3_to_local(self):
        """Original artifact locations are s3 paths. This updates artifact
        paths to local paths

        Modifies dataset inplace
        """
        artifact_destination = Path(self._artifact_destination)

        for i in range(len(self._dataset)):
            artifact = self._dataset[i]

            for artifact_type in ('max', 'avg', 'mask'):
                path = artifact_destination / \
                       f'{artifact_type}_{artifact.roi_id}.png'
                if not path.exists():
                    raise RuntimeError(f'{path} does not exist')
            artifact = Artifact(
                experiment_id=artifact.experiment_id,
                roi_id=artifact.roi_id,
                max_projection_path=(artifact_destination /
                                     f'max_{artifact.roi_id}.png'),
                avg_projection_path=(artifact_destination /
                                     f'avg_{artifact.roi_id}.png'),
                mask_path=(artifact_destination / f'mask_'
                                                  f'{artifact.roi_id}.png'),
                label=artifact.label
            )
            self._dataset[i] = artifact
