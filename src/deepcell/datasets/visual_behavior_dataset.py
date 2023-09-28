import glob
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Generator, Dict, Tuple, Set, Iterable, Union
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd

try:
    from croissant.utils import read_jsonlines, json_load_local_or_s3
except ModuleNotFoundError:
    pass
from tqdm import tqdm

from deepcell.datasets.model_input import ModelInput

logging.basicConfig(level=logging.INFO)

S3_MANIFEST_PREFIX = 'visual_behavior/manifests'

# The minimum global ROI id that was used by SLAPP/sagemaker
GLOBAL_ROI_ID_MIN = 1e6

# These labeling projects were added after first week of december 2020 and
# were found to hurt performance
EXCLUDE_PROJECTS = [
    'ophys-experts-slc-oct-2020_ophys-experts-go-big-or-go-home',
    'ophys-expert-danielsf-additions'
]


class VisualBehaviorDataset:
    """Class representing the visual behavior dataset from AWS sagemaker
    labeling jobs"""
    def __init__(self, artifact_destination: Path,
                 exclude_projects: Optional[List[str]] = None,
                 s3_manifest_prefix: str = S3_MANIFEST_PREFIX,
                 download=True,
                 debug=False):
        """
        Args:
            - artifact_destination
                Where to download the artifacts to
            - exclude_projects:
                an optional list of project names to exclude from being
                included in the dataset
            - s3_manifest_prefix
                Prefix on s3 where labeling manifests are stored
            - download
                Whether to download files from S3
            - debug
                If True, will only download a few files

        """
        if exclude_projects is None:
            exclude_projects = EXCLUDE_PROJECTS

        self._logger = logging.getLogger(__name__)
        self._exclude_projects = exclude_projects
        self._s3_manifest_prefix = s3_manifest_prefix
        self._artifact_destination = artifact_destination

        # Due to historic reasons, some ROI ids are stored in the manifest
        # using a globally unique ID
        # This object maps from global to an experiment-local id and vice versa
        self._global_to_local_map = json_load_local_or_s3(
            uri='s3://prod.slapp.alleninstitute.org/visual_behavior'
                '/global_to_local_mapping.json')

        self._logger.info('Reading manifests and preprocessing')
        self._dataset, artifact_dirs, self._project_meta = self._get_dataset()

        self._debug = debug

        os.makedirs(self._artifact_destination, exist_ok=True)

        if debug:
            not_cell_idx = np.argwhere([
                x.label != 'cell' for x in self._dataset])[:2]\
                .squeeze().tolist()
            cell_idx = np.argwhere([
                x.label == 'cell' for x in self._dataset])[:2]\
                .squeeze().tolist()
            self._dataset = [
                x for i, x in enumerate(self._dataset)
                if i in not_cell_idx + cell_idx
            ]

        self._update_artifact_locations_from_s3_to_local()

        if debug:
            self._download_files()
        else:
            is_already_downloaded = self._are_files_already_downloaded()
            if download:
                if not is_already_downloaded:
                    self._logger.info('Downloading dataset')
                    self._download_dataset(artifact_dirs=artifact_dirs)

                    self._logger.info('Renaming artifacts')
                    self._rename_artifacts()
                else:
                    self._logger.info('Dataset already downloaded...')
                self._validate_all_files_exist()

    @property
    def dataset(self) -> List[ModelInput]:
        return self._dataset

    @property
    def project_meta(self) -> pd.DataFrame:
        return self._project_meta

    def _get_dataset(self) -> \
            Tuple[List[ModelInput], Set[str], pd.DataFrame]:
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

        def is_failed(x: dict):
            """Checks whether this record is a failed record"""
            metadata_key = [x for x in x.keys() if 'metadata' in x]
            if len(metadata_key) == 1:
                if 'failure-reason' in x[metadata_key[0]]:
                    return True
            return False

        def get_roi_id(x: dict) -> str:
            """Parses ROI id and converts to experiment-local"""
            roi_id = x['roi-id']
            try:
                roi_id = int(roi_id)
            except ValueError:
                # ie 'exp_920211809_226'
                roi_id = roi_id.split('_')[-1]
                roi_id = int(roi_id)

            if roi_id >= GLOBAL_ROI_ID_MIN:
                # convert to experiment-local
                roi_id = self._global_to_local_map['global_to_local'][str(
                    roi_id)]
                roi_id = roi_id['roi_id']

            roi_id = f'exp_{experiment_id}_roi_{roi_id}'
            return roi_id

        def is_duplicate(global_to_local_map: dict, roi_id: str, seen: set) \
                -> bool:
            """Check if roi id is duplicate. If duplicate, it is either
            listed as a degeneracy in the global_to_local_map or has already
            been seen"""
            global_id = global_to_local_map[
                'local_to_global'].get(roi_id, None)
            if global_id is not None:
                if global_id in global_to_local_map['degeneracies']:
                    return True

            if roi_id in seen:
                return True

            return False

        def get_project_name(x: dict) -> Optional[str]:
            """Get project name by checking for a dict with key
            'workerAnnotations or majorityLabel'"""
            project_name = None
            for key in x:
                if isinstance(x[key], dict):
                    if 'workerAnnotations' in x[key] or 'majorityLabel' in \
                            x[key]:
                        return key
            return project_name

        manifests = self._get_manifests()
        rois = []
        artifact_dirs = set()
        seen = set()
        project_name_meta = []
        for file, manifest in manifests.items():
            for x in manifest:
                if self._exclude_projects is not None:
                    if is_exclude_project(x=x):
                        continue

                if is_failed(x=x):
                    continue

                experiment_id = x['experiment-id']
                roi_id = get_roi_id(x=x)
                x['roi-id'] = roi_id

                if is_duplicate(
                        roi_id=roi_id,
                        global_to_local_map=self._global_to_local_map,
                        seen=seen):
                    continue

                project_name = get_project_name(x=x)

                label = x[project_name].get('majorityLabel', None)
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
                    x['roi-mask-source-ref']).group()
                artifact_dirs.add(artifact_dir)

                artifact = ModelInput(
                    roi_id=x['roi-id'],
                    experiment_id=x['experiment-id'],
                    max_projection_path=x['max-source-ref'],
                    avg_projection_path=x['avg-source-ref'],
                    mask_path=x['roi-mask-source-ref'],
                    project_name=project_name,
                    label=label)
                rois.append(artifact)
                seen.add(roi_id)
        return rois, artifact_dirs, pd.DataFrame(project_name_meta)

    def _rename_artifacts(self):
        """
        Conform artifacts to exp_<experiment_id>_roi_<roi_id> convention
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
            if int(original_id) >= GLOBAL_ROI_ID_MIN:
                exp_id = self._global_to_local_map['global_to_local'][
                    original_id]['experiment_id']
                roi_id = self._global_to_local_map['global_to_local'][
                    original_id]['roi_id']
            else:
                # exp_<exp_id>_<roi_id>
                exp_id = re.findall(r'\d+', file.stem)[0]
                roi_id = original_id

            try:
                int(exp_id)
            except ValueError:
                raise RuntimeError(f'Invalid exp_id: {exp_id}')
            
            try:
                int(roi_id)
            except ValueError:
                raise RuntimeError(f'Invalid roi_id: {roi_id}')

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
            self._logger.info(f'Downloading from {bucket}')

            with tempfile.TemporaryDirectory() as d:
                cmd = f'aws s3 sync {bucket} {d} --exclude "*" ' \
                      f'--include "*.png" --quiet'
                os.system(cmd)

                # s3 sync downloads files include directory structure
                # Remove directory structure and just copy files to destination
                self._logger.info('Cleaning up directory structure. Moving '
                                  'files to artifact_destination')
                files = glob.glob(f'{d}/*/*.png') + glob.glob(f'{d}/*.png')
                for file in files:
                    if (self._artifact_destination / Path(file).name).exists():
                        continue
                    shutil.move(file, self._artifact_destination)

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

    def _get_manifests(self) -> Dict[str, Generator[Dict, None, None]]:
        """
        Each labeling job on AWS sagemaker produces a manifest
        This function retrieves all manifests for visual behavior labeling jobs

        Returns:
            List of generator of manifest records
        """
        s3 = boto3.client('s3')
        bucket = 'prod.slapp.alleninstitute.org'

        prefix = self._s3_manifest_prefix
        objs = s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix)
        files = [obj['Key'] for obj in objs['Contents'] if
                 obj['Key'] != prefix]
        manifests = {file: read_jsonlines(uri=f's3://{bucket}/{file}') for
                     file in files}
        return manifests

    def _update_artifact_locations_from_s3_to_local(self):
        """Original artifact locations are s3 paths. This updates artifact
        paths to local paths

        Modifies dataset inplace
        """
        artifact_destination = Path(self._artifact_destination)

        for i in range(len(self._dataset)):
            artifact = self._dataset[i]

            correlation_projection_path = (artifact_destination /
                                           f'corr_{artifact.roi_id}.png')
            if not correlation_projection_path.exists():
                # Due to historical reasons, we cannot produce correlation
                # projection for all inputs
                correlation_projection_path = None

            artifact = ModelInput(
                experiment_id=artifact.experiment_id,
                roi_id=artifact.roi_id,
                max_projection_path=(artifact_destination /
                                     f'max_{artifact.roi_id}.png'),
                avg_projection_path=(artifact_destination /
                                     f'avg_{artifact.roi_id}.png'),
                correlation_projection_path=correlation_projection_path,
                mask_path=(artifact_destination / f'mask_'
                                                  f'{artifact.roi_id}.png'),
                label=artifact.label,
                project_name=artifact.project_name
            )
            self._dataset[i] = artifact

    def _are_files_already_downloaded(self):
        """Returns True if all files already exist at destination"""
        for roi in self._dataset:
            if not roi.avg_projection_path.exists():
                return False
            if not roi.max_projection_path.exists():
                return False
            if not roi.mask_path.exists():
                return False
        return True

    def _validate_all_files_exist(self):
        """Validates that all files exist"""
        for artifact in self._dataset:
            for artifact_type in ('max', 'avg', 'mask'):
                path = self._artifact_destination / \
                       f'{artifact_type}_{artifact.roi_id}.png'
                if not path.exists():
                    raise RuntimeError(f'{path} does not exist')

