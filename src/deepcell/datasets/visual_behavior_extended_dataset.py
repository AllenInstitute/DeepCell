import datetime
from pathlib import Path
from typing import Optional, List, Generator, Dict

from deepcell.datasets.visual_behavior_dataset import VisualBehaviorDataset


class VisualBehaviorExtendedDataset(VisualBehaviorDataset):
    def __init__(self, artifact_destination: Path,
                 exclude_projects: Optional[List[str]] = None,
                 download=True,
                 debug=False):
        """Represents the visual behavior dataset extended with additional
        data"""
        super().__init__(artifact_destination=artifact_destination,
                         exclude_projects=exclude_projects,
                         s3_manifest_prefix='visual_behavior_extended'
                                            '/manifests/',
                         download=download,
                         debug=debug)

    def _get_manifests(self) -> Dict[str, Generator[Dict, None, None]]:
        manifests = super()._get_manifests()

        def update_manifest(manifest,
                            name: str,
                            label: str,
                            creation_date: datetime):
            """
            Adds key <name> which contains label, as well as creation date
            Args:
                manifest:
                    The manifest to add to
                name:
                    Name of the "project"
                label:
                    Label to assign to each observation
                creation_date:
                    Date this data was compiled

            Returns:

            """
            new_manifest = []
            for x in manifest:
                x[name] = {
                    # These ROIs are all not cell
                    'majorityLabel': label
                }

                x[f'{name}-metadata'] = {
                    'creation-date': creation_date
                }
                new_manifest.append(x)

            for x in new_manifest:
                yield x

        for i, (file, manifest) in enumerate(manifests.items()):
            file_name = Path(file).name
            if file_name == 'vasculature.manifest':
                manifest = update_manifest(manifest=manifest,
                                           name='add_vasculature',
                                           creation_date=datetime.datetime(
                                             year=2021, month=11, day=11),
                                           label='not cell')
                manifests[file] = manifest
            elif file_name == 'large_processes_manifest':
                manifest = update_manifest(manifest=manifest,
                                           name='large_processes',
                                           creation_date=datetime.datetime(
                                             year=2021, month=11, day=22),
                                           label='cell')
                manifests[file] = manifest
        return manifests
